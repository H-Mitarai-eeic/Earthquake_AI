import {
    Box,
    CircularProgress,
    Grid,
    Slider,
    Stack,
    styled,
    Typography,
    useTheme,
} from "@mui/material";
import { Dispatch, FC, SetStateAction, useState } from "react";
import { BITSIZE, DEFAULT_DATA, SCALE } from "../global";
import JapanMap from "./JapanMap";
import ObservePoint from "./ObservePoint";

const Container = styled(Stack)(({ theme }) => ({
    margin: theme.spacing(2),
}));

type Props = {
    pos: number[];
    setPos: Dispatch<SetStateAction<number[]>>;
    data: number[];
    setData: Dispatch<SetStateAction<number[]>>;
    isLoading: boolean;
};

const ObservePoints: FC<{ pos: number[]; data: number[] }> = ({
    pos,
    data,
}) => {
    let dataToMap = [];
    for (let w = 0; w < BITSIZE; w++) {
        for (let h = 0; h < BITSIZE; h++) {
            dataToMap.push({ x: w, y: h, value: data[h * BITSIZE + w] });
        }
    }
    const items: JSX.Element[] = [];
    dataToMap.map((v, index) => {
        items.push(
            <ObservePoint
                key={index}
                x={v.x}
                y={v.y}
                value={v.value}
                isSelected={v.x == pos[0] && v.y == pos[1]}
            />
        );
    });
    return <>{items}</>;
};

const MapOfJapan: FC<Props> = ({ pos, setPos, data, setData, isLoading }) => {
    const theme = useTheme();
    return (
        <Container
            gap={theme.spacing(3)}
            direction={"column"}
            justifyContent={"center"}
        >
            <Box
                id="CANVAS"
                border={"1px solid"}
                // borderColor={"black"}
                sx={{ position: "relative" }}
                onClick={(event) => {
                    if (isLoading) return;
                    setData(DEFAULT_DATA);
                    let clickX = event.pageX;
                    let clickY = event.pageY;

                    let clientRect = document
                        ?.getElementById("CANVAS")
                        ?.getBoundingClientRect();
                    let positionX =
                        (clientRect?.left || 0) + window.pageXOffset;
                    let positionY = (clientRect?.top || 0) + window.pageYOffset;

                    let x = clickX - positionX;
                    let y = clickY - positionY;

                    let X = Math.round(x / (8 * SCALE));
                    let Y = Math.round(y / (9 * SCALE));
                    setPos([X, Y]);
                }}
            >
                <JapanMap
                    sx={{ position: "absolute", top: "0px", left: "0px" }}
                />
                <ObservePoints pos={pos} data={data} />
                {isLoading ? (
                    <CircularProgress
                        size={"200px"}
                        color="secondary"
                        sx={{
                            position: "absolute",
                            top: "200px",
                            left: "180px",
                            zIndex: "120",
                        }}
                    />
                ) : null}
            </Box>
        </Container>
    );
};

export default MapOfJapan;
