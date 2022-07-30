import {
    Box,
    Grid,
    Slider,
    Stack,
    styled,
    Typography,
    useTheme,
} from "@mui/material";
import { FC, useState } from "react";
import {
    BITSIZE,
    DEFAULT_DATA,
    LATITUDE_MAX,
    LATITUDE_SPAN,
    LONGTITUDE_MIN,
    LONGTITUDE_SPAN,
} from "../shared/global";
import InputBar from "../shared/InputBar";
import MapOfJapan from "../shared/MapOfJapan";
import RunButton from "../shared/RunButton";

const subContainer = styled(Stack)(({ theme }) => ({
    margin: theme.spacing(2),
}));

const English: FC = () => {
    const theme = useTheme();
    const [pos, setPos] = useState([32, 32]);
    const [mag, setMag] = useState(7.5);
    const [depth, setDepth] = useState(350);
    const [isLoading, setIsLoading] = useState(false);

    const [data, setData] = useState(DEFAULT_DATA);
    return (
        <Grid container>
            <Grid xs={12} lg={6}>
                <Stack alignItems={"center"} justifyItems={"center"} gap={1}>
                    <Typography
                        variant="h1"
                        textAlign={"left"}
                        marginBottom={2}
                    >
                        Earthquaker
                    </Typography>
                    <InputBar
                        title={"magnitude"}
                        min={5}
                        max={10}
                        step={0.1}
                        value={mag}
                        setValue={setMag}
                    />
                    <InputBar
                        title={"depth"}
                        min={0}
                        max={700}
                        step={1}
                        value={depth}
                        setValue={setDepth}
                    />
                    <Stack direction={"row"} gap={5}>
                        <Typography variant="h4">
                            latitude :&nbsp;
                            {Math.round(
                                LATITUDE_MAX -
                                    (pos[1] * LATITUDE_SPAN) / BITSIZE
                            )}
                            °N
                        </Typography>
                        <Typography variant="h4">
                            longtitude :&nbsp;
                            {Math.round(
                                LONGTITUDE_MIN +
                                    (pos[0] * LONGTITUDE_SPAN) / BITSIZE
                            )}
                            °E
                        </Typography>
                    </Stack>
                    <RunButton
                        x={pos[0]}
                        y={pos[1]}
                        mag={mag}
                        depth={depth}
                        setData={setData}
                        setIsLoading={setIsLoading}
                    />
                </Stack>
            </Grid>

            <Grid xs={12} lg={6}>
                <Stack
                    alignItems={"center"}
                    justifyItems={"center"}
                    direction={"column"}
                >
                    <MapOfJapan
                        pos={pos}
                        setPos={setPos}
                        data={data}
                        setData={setData}
                        isLoading={isLoading}
                    ></MapOfJapan>
                </Stack>
            </Grid>
        </Grid>
    );
};

export default English;
