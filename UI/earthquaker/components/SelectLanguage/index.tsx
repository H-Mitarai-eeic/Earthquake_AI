import {
    Box,
    Grid,
    Slider,
    Stack,
    styled,
    Typography,
    useTheme,
} from "@mui/material";
import { Dispatch, FC, SetStateAction, useState } from "react";
import LanguageIcon from "@mui/icons-material/Language";
import { Lang } from "../shared/global";

const Container = styled(Stack)(({ theme }) => ({
    margin: theme.spacing(2),
    width: "80%",
}));

type Props = {
    language: Lang;
    setLanguage: Dispatch<SetStateAction<Lang>>;
};

const SelectLanguage: FC<Props> = ({ language, setLanguage }) => {
    const nextLanguage: { [key: string]: Lang } = {
        English: "Japanese",
        Japanese: "English",
    };
    const display = { English: "日本語", Japanese: "English" };
    return (
        <Stack
            alignItems={"center"}
            onClick={() => {
                setLanguage(nextLanguage[language]);
            }}
        >
            <LanguageIcon fontSize="large"></LanguageIcon>
            <Typography variant="h6">{display[language]}</Typography>
        </Stack>
    );
};

export default SelectLanguage;
