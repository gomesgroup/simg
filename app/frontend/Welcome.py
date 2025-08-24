import streamlit as st


def app():
    st.set_page_config(layout="wide")
    css = '''
    <style>
        section.stMain > div {max-width:1000px}
        #stDecoration {background-color: #C41230; background-image: none;}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    cols = st.columns(3)
    cols[0].image('cmu.png')

    cols = st.columns([0.8, 0.2])
    cols[0].header(
        """Advancing Molecular Machine Learning Representations with Stereoelectronics-Infused Molecular Graphs""")
    cols[0].write("by Daniil A Boiko, Thiago Reschützegger, Benjamin Sanchez-Lengeling, Samuel M Blau, Gabe Gomes")
    cols[0].write("")

    if st.button("Submit"):
        st.write("Redirecting to the submit page")
        st.switch_page("pages/1_Submit.py")

    cols = st.columns([0.5, 0.5])
    cols[0].write("""
            This application allows for generation of stereoelectronics-infused molecular graphs from either SMILES strings (one conformer will be generated) or .xyz files with coordinates of all atoms. Resulting outputs include extended molecular graphs, predicted atomic charges, lone pair characteristics, and orbital interactions among others. To use the service, perform the following steps:
            1. Click “Submit” above.
            2. Draw your structure (or paste a SMILES string, which can be copied from an editor of choice) or upload a .xyz file. Alternatively, you can select one of the examples above.
            3. You will be transferred to a new screen to explore these interactions.
            """)

    cols[1].image('screenshot.png')


app()
