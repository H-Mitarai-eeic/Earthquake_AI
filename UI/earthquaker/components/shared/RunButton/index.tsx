import { Box, styled, Typography, useTheme } from "@mui/material";
import Link from "next/link";
import { Dispatch, FC, SetStateAction, useState } from "react";
import theme from "../../../styles/theme";
import axios from "axios";

const Container = styled(Box)(({ theme }) => ({
    borderRadius: "50%",
    textAlign: "center",
    margin: theme.spacing(5),
    width: "20vmin",
    height: "20vmin",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    transitionDuration: ".2s",
    backgroundColor: theme.palette.primary.main,
    boxShadow: `0px 0px 10px ${theme.palette.secondary.main}`,
    "&:hover": {
        boxShadow: `0px 0px 50px ${theme.palette.secondary.main}`,
    },
}));

const run = (
    x: number,
    y: number,
    mag: number,
    depth: number,
    setData: Dispatch<SetStateAction<number[]>>,
    setIsLoading: Dispatch<SetStateAction<boolean>>
) => {
    console.warn({ x }, { y }, { mag }, { depth });
    setIsLoading(true);
    const URL =
        "http://localhost:8000?x=" +
        x +
        "&y=" +
        y +
        "&mag=" +
        mag +
        "&depth=" +
        depth;
    axios.get(URL).then((response) => {
        console.log("got", response.data);
        setIsLoading(false);
        setData(response.data.split(","));
    });
};

export type Props = {
    x: number;
    y: number;
    mag: number;
    depth: number;
    setData: Dispatch<SetStateAction<number[]>>;
    isLoading: boolean;
    setIsLoading: Dispatch<SetStateAction<boolean>>;
};

export const RunButton: FC<Props> = ({
    x,
    y,
    mag,
    depth,
    setData,
    isLoading,
    setIsLoading,
}) => {
    return (
        <Container
            onClick={() => {
                if (isLoading) return;
                run(x, y, mag, depth, setData, setIsLoading);
            }}
            sx={{
                boxShadow: isLoading
                    ? `0px 0px 10px ${theme.palette.primary.dark}`
                    : `0px 0px 10px ${theme.palette.secondary.main}`,

                "&:hover": {
                    boxShadow: isLoading
                        ? `0px 0px 10px ${theme.palette.primary.dark}`
                        : `0px 0px 50px ${theme.palette.secondary.main}`,
                },
            }}
        >
            <Typography variant="h4" color={"white"}>
                {isLoading ? "RUNNING" : "RUN"}
            </Typography>
        </Container>
    );
};

export default RunButton;
