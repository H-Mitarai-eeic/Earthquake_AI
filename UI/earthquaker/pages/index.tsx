import { Box, Stack, styled, Typography, useTheme } from "@mui/material";
import type { NextPage } from "next";
import { useState } from "react";
import Earthquaker from "../components/Earthquaker";
import English from "../components/Earthquaker";
import Layout from "../components/Layout";
import SelectLanguage from "../components/SelectLanguage";
import { Lang } from "../components/shared/global";
import StarIcon from "@mui/icons-material/Star";
const Home: NextPage = () => {
    const [language, setLanguage] = useState<Lang>("English");
    const [modalOpen, setModalOpen] = useState(false);

    const theme = useTheme();
    return (
        <Layout title={"Earthquaker"}>
            <Stack
                direction="row"
                justifyContent="flex-end"
                alignItems="center"
                spacing={4}
                marginBottom={theme.spacing(2)}
            >
                <Stack
                    alignItems={"center"}
                    justifyContent={"space-between"}
                    onClick={() => {
                        setModalOpen(true);
                    }}
                >
                    <StarIcon fontSize="large"></StarIcon>
                    <Typography variant="h6">presets</Typography>
                </Stack>
                <SelectLanguage
                    language={language}
                    setLanguage={setLanguage}
                ></SelectLanguage>
            </Stack>
            <Earthquaker
                language={language}
                modalOpen={modalOpen}
                setModalOpen={setModalOpen}
            ></Earthquaker>
        </Layout>
    );
};

export default Home;
