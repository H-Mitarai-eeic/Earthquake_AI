import { Box, styled, Typography, useTheme } from "@mui/material";
import Link from "next/link";
import { FC } from "react";
import theme from "../../../styles/theme";

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

const run = (x: number, y: number, mag: number, depth: number) => {
    console.warn({ x }, { y }, { mag }, { depth });
};

export type Props = {
    x: number;
    y: number;
    mag: number;
    depth: number;
};

export const RunButton: FC<Props> = ({ x, y, mag, depth }) => {
    return (
        <Container
            onClick={() => {
                run(x, y, mag, depth);
            }}
        >
            <Typography variant="h4" color={"white"}>
                RUN
            </Typography>
        </Container>
    );
};

export default RunButton;
