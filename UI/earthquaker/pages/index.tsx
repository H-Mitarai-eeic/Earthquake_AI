import { Typography, useTheme } from "@mui/material";
import type { NextPage } from "next";
import English from "../components/English";
import Layout from "../components/Layout";
import styles from "../styles/Home.module.css";

const Home: NextPage = () => {
    return (
        <Layout title={"Earthquaker"}>
            <English></English>
        </Layout>
    );
};

export default Home;
