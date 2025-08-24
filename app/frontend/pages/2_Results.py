import os
import time

import streamlit as st
from streamlit_molstar import st_molstar_content

import redis
from rq import Queue

import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
css = '''
<style>
    section.stMain > div {max-width:1200px}
    #stDecoration {background-color: #C41230; background-image: none;}
</style>
'''
st.markdown(css, unsafe_allow_html=True)


def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)


def get_number_df(job):
    atom_targets = ['Charge', 'Core', 'Valence', 'Total']
    bond_targets = ['occupancy', 's', 'p', 'd', 'f', 'pol_diff', 'pol_coeff_diff']
    lone_pair_targets = ['s', 'p', 'd', 'f', 'occupancy']

    columns = [f"atom/{target}" for target in atom_targets] + \
              [f"lone_pair/{target}" for target in lone_pair_targets] + \
              [f"bond/{target}" for target in bond_targets]

    numbers = pd.DataFrame(
        job.result['node_level'],
        columns=columns
    ).map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    numbers.loc[job.result['is_atom'], [f"lone_pair/{target}" for target in lone_pair_targets]] = ''
    numbers.loc[job.result['is_atom'], [f"bond/{target}" for target in bond_targets]] = ''
    numbers.loc[job.result['is_lp'], [f"atom/{target}" for target in atom_targets]] = ''
    numbers.loc[job.result['is_lp'], [f"bond/{target}" for target in bond_targets]] = ''
    numbers.loc[job.result['is_bond'], [f"atom/{target}" for target in atom_targets]] = ''
    numbers.loc[job.result['is_bond'], [f"lone_pair/{target}" for target in lone_pair_targets]] = ''

    return numbers


if 'job_id' not in st.query_params:
    st.switch_page("pages/1_Submit.py")

redis_url = os.environ.get('REDIS_HOST')
conn = redis.from_url(redis_url)

q = Queue(connection=conn)

job_id = st.query_params['job_id']
job = q.fetch_job(job_id)

if job.meta['status'] == 'Completed' and job.result:
    numbers = get_number_df(job)

    cols = st.columns(2)
    cols[0].header("Prediction results", divider='gray')
    cols[1].warning(
        "The primary goal of this model is to provide a feature rich representation for molecular ML. The model can make mistakes; please, carefully validate the results.",
        icon="⚠️"
    )

    st.write('Generate or provided geometry')
    st.info(
        'This is the geometry used as input. If you provided a SMILES string, the geometry was generated using Open Babel.',
        icon='ℹ️')
    st_molstar_content(
        job.result['xyz'],
        'xyz',
    )

    cols = st.columns(2)

    with cols[0]:
        st.write('Generated extended input graph')
        st.info(
            'This graph is used as an input to the second model in the pipeline to populate the graph with NBO features',
            icon='ℹ️')
        st.image(
            job.result['graph'],
            use_container_width=True
        )

        st.write("Lone pair targets")
        st.info(
            "The lone pair targets are the predicted NBO features. Their indexes refer to node indexes in the top left graph.",
            icon='ℹ️')
        lp_subset = numbers[numbers['lone_pair/s'] != ''][
            ['lone_pair/s', 'lone_pair/p', 'lone_pair/occupancy']
        ]
        lp_subset.columns = ['s, %', 'p, %', 'occupancy']
        st.table(lp_subset)

        st.write("Bond targets")
        st.info(
            "The bond targets are the predicted NBO features. Their indexes refer to node indexes in the top left graph.",
            icon='ℹ️')
        st.warning(
            "Some values can larger than 2, it does not mean that the bond has more than two electrons, but the fact that electrons are donated to corresponding orbitals.",
            icon="⚠️")
        bond_subset = numbers[numbers['bond/occupancy'] != ''][['bond/occupancy']]
        bond_subset.columns = ['occupancy']
        bond_subset = bond_subset.reset_index().rename(columns={'index': 'bond'}).set_index('bond')

        a2b_results = pd.DataFrame(np.hstack(
            [job.result['a2b_index'].T[:, ::-1], job.result['a2b_results']]
        ), columns=['bond', 'atom', 's', 'p', 'd', 'f', 'pol', 'pol_coeff', ]).set_index('bond')
        a2b_results[['s', 'p', 'd', 'f']] = a2b_results[['s', 'p', 'd', 'f']].clip(0, 1)
        a2b_results[['s', 'p', 'd', 'f']] /= a2b_results[['s', 'p', 'd', 'f']].sum(axis=1).values[:, None]

        a2b_results = a2b_results.join(bond_subset)

        st.dataframe(a2b_results)

    with cols[1]:
        st.write('Atom charges')
        st.info(
            'The atom charges are the predicted NBO features. Their indexes refer to atom indexes in the top left graph.',
            icon='ℹ️')
        st.image(
            job.result['atom_graph'],
            use_container_width=True
        )

        st.write("Atom targets")
        st.info(
            "The atom targets are the predicted NBO features. Their indexes refer to node indexes in the top left graph.",
            icon='ℹ️')
        atom_subset = numbers[numbers['atom/Charge'] != ''][
            ['atom/Charge', 'atom/Core', 'atom/Valence', 'atom/Total']
        ]

        atom_subset.columns = ['Charge', 'Core', 'Valence', 'Total']
        st.table(atom_subset)

    # disclaimer about number of electrons

    st.divider()

    st.write("Interactions")
    cols = st.columns(2)

    with cols[0]:
        st.image(
            job.result['interactions'],
            caption="Interaction matrix",
            use_container_width=True
        )

    with cols[1]:
        st.info("The interaction matrix shows the probability of interaction between two nodes in the graph.",
                icon='ℹ️')
        st.warning(
            "This does not reflect strength of this interaction, only probability of it appearing in NBO program output.",
            icon="⚠️")
        st.write("Explore specific interaction")
        n_atoms = job.result['is_atom'].sum()

        subcols = st.columns(2)
        with subcols[0]:
            interaction_a = st.selectbox(
                "Interaction A",
                np.arange(len(job.result['interaction_table'])) + n_atoms,
                index=None
            )
            interaction_b = st.selectbox(
                "Interaction B",
                np.arange(len(job.result['interaction_table'])) + n_atoms,
                index=None
            )

        if interaction_a and interaction_b:
            with subcols[1]:
                if interaction_a == interaction_b:
                    st.write("Cannot select the same node")
                else:
                    probability = job.result['interaction_table'][interaction_a - n_atoms, interaction_b - n_atoms]
                    st.write(f"Probability: {probability:.3f}")
                    feature_names = ["Perturbation energy", "Energy difference", "Fock matrix element"]
                    interaction_predictions = job.result['interaction_predictions'][
                        (interaction_a - n_atoms) * len(job.result['interaction_table']) + (interaction_b - n_atoms)
                        ]

                    interaction_predictions = pd.DataFrame(
                        [interaction_predictions],
                        columns=feature_names
                    ).T

                    interaction_predictions.loc['Perturbation energy'] *= 100

                    interaction_predictions.columns = ['Value']
                    st.table(interaction_predictions)

                    if probability < 0.9:
                        st.write('Interactions with low probabilities will have inaccurate predictions')

else:
    st.write(f"Status: {job.meta['status']}")
    st.write("Refreshing the page in one second")
    st.write("Please wait...")
    time.sleep(1)
    st.rerun()
