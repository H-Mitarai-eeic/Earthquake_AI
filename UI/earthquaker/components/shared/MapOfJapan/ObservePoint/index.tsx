import { Box, Typography, useTheme } from "@mui/material";
import { Dispatch, FC, SetStateAction, useState } from "react";
import { SCALE } from "../../global";

// const SCALE = 1.2;
const width = 8 * SCALE;
const height = 9 * SCALE;

const myColorList = [
    "rgb(119,194,229)",
    "rgb(129,239,125)",
    "rgb(255,228,9)",
    "rgb(254,148,6)",
    "rgb(251,0,6)",
    "rgb(251,0,6)",
    "rgb(251, 0, 255)",
    "rgb(107, 0, 109)",
    "rgb(107, 0, 109)",
];

const fontColor = [
    "black",
    "black",
    "black",
    "black",
    "white",
    "white",
    "white",
    "white",
    "white",
];

type Props = {
    x: number;
    y: number;
    value: number;
    // setValue?: Dispatch<SetStateAction<number>>;
};

const ObservePoint: FC<Props> = ({ x, y, value }) => {
    const theme = useTheme();

    if (value == 0) {
        return null;
    }

    return (
        <Box
            sx={{
                width: width,
                height: height,
                borderRadius: "50%",
                opacity: "0.9",
                position: "absolute",
                left: `${x * width - width}px`,
                top: `${y * height}px`,
                backgroundColor: `${myColorList[value - 1]}`,
                color: `${fontColor[value - 1]}`,
            }}
        >
            <Typography
                fontSize={"5px"}
                textAlign={"center"}
                lineHeight={`${height}px`}
            >
                {value}
            </Typography>
        </Box>
    );
};

export default ObservePoint;
