import { Box, Typography, useTheme } from "@mui/material";
import { Dispatch, FC, SetStateAction, useState } from "react";
import { SCALE } from "../../global";
import PanoramaFishEyeIcon from "@mui/icons-material/PanoramaFishEye";
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
    isSelected?: boolean;
};

const ObservePoint: FC<Props> = ({ x, y, value, isSelected }) => {
    const theme = useTheme();

    if (value == 0 && !isSelected) {
        return null;
    }

    return isSelected ? (
        <>
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
                {value != 0 ? (
                    <Typography
                        fontSize={"5px"}
                        textAlign={"center"}
                        lineHeight={`${height}px`}
                    >
                        {value}
                    </Typography>
                ) : null}
            </Box>
            <Box
                sx={{
                    width: 5 * width,
                    height: 5 * height,
                    borderRadius: "50%",
                    position: "absolute",
                    left: `${x * width - 3 * width}px`,
                    top: `${y * height - height}px`,
                    color: "red",
                    zIndex: 100,
                }}
            >
                <Typography
                    fontSize={"10px"}
                    textAlign={"center"}
                    lineHeight={`${height}px`}
                >
                    <PanoramaFishEyeIcon fontSize="large" />
                </Typography>
            </Box>
        </>
    ) : (
        <Box
            sx={{
                width: width,
                height: height,
                borderRadius: "50%",
                opacity: "0.8",
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
