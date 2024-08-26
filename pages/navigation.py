import streamlit as st


def navigation():
    with st.sidebar:

        # logo = "logo.png"
        # st.logo(logo)

        # This markdown is used for the logo image
        # st.markdown(
        #     """
        #     <style>
        #         div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
        #             height: 4rem;
        #             width: auto;
        #         }
        #     </style>
        #     """,
        #     unsafe_allow_html=True,
        # )

        # This markdown is used for the title text
        st.markdown(
            """
            <style>
                .title{
                    color: #752303;
                    font-size: 33px;
                }
            </style>

            <p class="title"> IMU Data </p>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.write("")

        st.page_link(
            "pages/index.py",
            label="IMU Data"
        )

        st.page_link(
            "pages/calculation.py",
            label="Angle Calculation"
        )

        st.button("Report")

        st.button("Log out")

    return True
