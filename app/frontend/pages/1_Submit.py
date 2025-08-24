import os

import streamlit as st
from streamlit_ketcher import st_ketcher
from streamlit_molstar import st_molstar_content

import redis
from rq import Queue

from rdkit import Chem

st.set_page_config(layout="wide")
css = '''
<style>
    section.stMain > div {max-width:1000px}
    #stDecoration {background-color: #C41230; background-image: none;}
</style>
'''
st.markdown(css, unsafe_allow_html=True)


def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

st.write('Try one of these molecules:')
cols = st.columns(3)
for i in range(3):
    with open(f'test_compounds/{i}.xyz', 'r') as f:
        xyz = f.read()

    with cols[i]:
        st_molstar_content(
            xyz,
            'xyz'
        )

        if st.button('Try', use_container_width=True, key=f'try_{i}'):
            redis_url = os.environ.get('REDIS_HOST')
            conn = redis.from_url(redis_url)

            q = Queue(connection=conn)
            job = q.enqueue("worker.submit_request", xyz)
            job.meta['status'] = 'Submitted'
            job.save_meta()

            st.write("Redirecting to the results page")
            nav_to(f"/Results?job_id={job.id}")

st.divider()
st.write('Provide your own:')
param = st_ketcher()

if param:
    violated = False

    if not violated:
        if '.' in param:
            violated = True
            st.write('Multiple molecules are not supported')

    if not violated:
        try:
            mol = Chem.MolFromSmiles(param)
        except:
            violated = True
            st.write("Invalid SMILES string")

    if not violated:
        # total charge
        total_charge = 0
        total_radical = 0

        for atom in mol.GetAtoms():
            total_charge += atom.GetFormalCharge()
            total_radical += atom.GetNumRadicalElectrons()

        if total_charge != 0:
            violated = True
            st.write("Total charge is not zero")

        if total_radical != 0:
            violated = True
            st.write("Not closed-shell")

    if not violated:
        redis_url = os.environ.get('REDIS_HOST')
        conn = redis.from_url(redis_url)

        q = Queue(connection=conn)
        job = q.enqueue("worker.submit_request", param)
        job.meta['status'] = 'Submitted'
        job.save_meta()

        st.write("Redirecting to the results page")
        nav_to(f"/Results?job_id={job.id}")

st.divider()
uploaded_file = st.file_uploader("... or upload a file")

if uploaded_file is not None:
    st.write("File uploaded; submitting the long request")

    contents = uploaded_file.read().decode()

    redis_url = os.environ.get('REDIS_HOST')
    conn = redis.from_url(redis_url)

    q = Queue(connection=conn)
    job = q.enqueue("worker.submit_request", contents)
    job.meta['status'] = 'Submitted'
    job.save_meta()

    st.write("Redirecting to the results page")
    nav_to(f"/Results?job_id={job.id}")
