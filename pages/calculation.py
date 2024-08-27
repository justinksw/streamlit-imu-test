import streamlit as st

from pages.navigation import navigation

st.set_page_config(layout="wide")

navigation()

rotation = st.session_state["rotation"]

container = st.container(height=None, border=True)

with container:

    col1, col2 = st.columns(
        [0.4, 0.6],
        gap="large",
        vertical_alignment="center"
    )

    with col1:

        st.image("img.png")

    with col2:

        st.subheader("Rotation")

        st.line_chart(
            rotation,
            x="index",
            y=["X", "Y", "Z"]
        )
