import numpy as np
import pandas as pd

from transforms3d import euler
from transforms3d import quaternions as quat

import streamlit as st

from pages.navigation import navigation

st.set_page_config(layout="wide")

navigation()


D2R = np.pi / 180  # degree_to_radian
R2D = 180 / np.pi  # radian_to_degree


class IMU:  # each device
    def __init__(self, device, df) -> None:

        self.device = device
        self.df = df

    def get_quaternion(self):

        return self.df["Quaternion"].to_numpy()


class IMUs:
    def __init__(self, file_path) -> None:

        df = pd.read_csv(file_path, index_col=False)

        # From the original df, Rename the columns

        df = df.rename(
            columns={
                "设备名称": "DeviceName",
                "角度X(°)": "EulerX",
                "角度Y(°)": "EulerY",
                "角度Z(°)": "EulerZ",
                "加速度X(g)": "AccelerationX",
                "加速度Y(g)": "AccelerationY",
                "加速度Z(g)": "AccelerationZ",
                "角速度X(°/s)": "AngularVelX",
                "角速度Y(°/s)": "AngularVelY",
                "角速度Z(°/s)": "AngularVelZ",
                "磁场X(ʯt)": "MagneticX",
                "磁场Y(ʯt)": "MagneticY",
                "磁场Z(ʯt)": "MagneticZ",
            }
        )

        # Select columns

        df_ = df[[
            "DeviceName",
            "EulerX",
            "EulerY",
            "EulerZ",
            "AccelerationX",
            "AccelerationY",
            "AccelerationZ",
            "AngularVelX",
            "AngularVelY",
            "AngularVelZ",
            "MagneticX",
            "MagneticY",
            "MagneticZ",
        ]]

        self.dff = df_.copy()

        self.dff = self.dff.reset_index()

        # Calculate the Quaternion based on Euler angles

        self.dff["Quaternion"] = self.dff.apply(
            lambda row: euler.euler2quat(
                row.EulerX*D2R,
                row.EulerY*D2R,
                row.EulerZ*D2R, 'sxyz'),
            axis=1
        )

        # Split the entire dataframe into devices

        device_names = [
            "WT5500002557_UDP",
            "WT5500002564_UDP",
        ]

        self.imus = []

        for name in device_names:

            dff_ = pd.DataFrame()
            dff_ = self.dff.loc[self.dff["DeviceName"] == name]

            self.imus.append(IMU(name, dff_))

    def get_imus_list(self):

        return self.imus

    def get_entire_df(self):

        return self.dff


class Calculation_Two_IMUs:
    def __init__(self, imu1, imu2) -> None:

        self.imu1 = imu1
        self.imu2 = imu2

    def get_rotation(self):

        q1 = self.imu1.get_quaternion()
        q2 = self.imu2.get_quaternion()

        rotation = []

        for i in range(len(q1)):
            rotation_q = quat.qmult(q2[i], quat.qinverse(q1[i]))

            rotation_euler = euler.quat2euler(rotation_q)

            rotation.append(rotation_euler)

        rotation = np.array(rotation) * R2D

        return rotation

    def get_rotation_df(self):
        data_ = self.get_rotation()

        # data_ = np.transpose(data)

        df = pd.DataFrame({
            "X": data_[:, 0],
            "Y": data_[:, 1],
            "Z": data_[:, 2],
        })

        df_ = df.reset_index()

        return df_


# def plotly_line_chart(df):
#     fig = px.line(
#         df,
#         x="index",
#         y=["AccelerationX", "AccelerationY", "AccelerationZ"]
#     )

#     fig.update_layout(
#         legend=dict(
#             yanchor="top",
#             y=5,
#             xanchor="left",
#             x=0.01
#         )
#     )

#     return fig


def index():

    file = IMUs("./data/data_0.csv")

    imus = file.get_imus_list()
    # imus[0].plot_euler()
    # imus[1].plot_euler()

    # imus = file.get_entire_df()

    st.title("IMU Data")

    st.selectbox(
        label="Select an IMU",
        options=["IMU-1", "IMU-2"],
    )

    container = st.container(height=None, border=True)

    with container:

        col1, col2 = st.columns(2, gap="large",)

        with col1:

            st.subheader("Accelerometer")

            st.line_chart(
                imus[0].df,
                x="index",
                y=["AccelerationX", "AccelerationY", "AccelerationZ"]
            )

            st.subheader("Gyroscope")

            st.line_chart(
                imus[0].df,
                x="index",
                y=["AngularVelX", "AngularVelY", "AngularVelZ"]
            )

        with col2:

            st.subheader("Magnetometer")

            st.line_chart(
                imus[0].df,
                x="index",
                y=["MagneticX", "MagneticY", "MagneticZ"]
            )

            st.subheader("Euler Angle")

            st.line_chart(
                imus[0].df,
                x="index",
                y=["EulerX", "EulerY", "EulerZ"]
            )

    rotation = Calculation_Two_IMUs(imus[0], imus[1]).get_rotation_df()

    st.session_state.rotation = rotation

    # rotation.plot_rotation()

    return None


index()
