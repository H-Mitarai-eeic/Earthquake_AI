import { Box, Stack, styled, Typography, useTheme } from "@mui/material";
import type { NextPage } from "next";
import { useState } from "react";
import Earthquaker from "../components/Earthquaker";
import English from "../components/Earthquaker";
import Layout from "../components/Layout";
import SelectLanguage from "../components/SelectLanguage";
import { Lang } from "../components/shared/global";

const Home: NextPage = () => {
    const [language, setLanguage] = useState<Lang>("English");
    const theme = useTheme();
    return (
        <Layout title={"Earthquaker"}>
            <Stack alignItems={"flex-end"} marginBottom={theme.spacing(2)}>
                <SelectLanguage
                    language={language}
                    setLanguage={setLanguage}
                ></SelectLanguage>
            </Stack>
            <Earthquaker language={language}></Earthquaker>
        </Layout>
    );
};

export default Home;
