import { Typography, useTheme } from "@mui/material";
import type { NextPage } from "next";
import { useState } from "react";
import Earthquaker from "../components/Earthquaker";
import English from "../components/Earthquaker";
import Layout from "../components/Layout";
import styles from "../styles/Home.module.css";

const Home: NextPage = () => {
    const [language, setLanguage] = useState("English");
    return (
        <Layout title={"Earthquaker"}>
            <Earthquaker language={"English"}></Earthquaker>
        </Layout>
    );
};

export default Home;
