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
import InputBar from "../shared/InputBar";
import MapOfJapan from "../shared/MapOfJapan";
import RunButton from "../shared/RunButton";

const subContainer = styled(Stack)(({ theme }) => ({
    margin: theme.spacing(2),
}));

const English: FC = () => {
    const theme = useTheme();
    const [x, setX] = useState(5);
    const [y, setY] = useState(5);
    const [mag, setMag] = useState(5);
    const [depth, setDepth] = useState(0);

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
                    <RunButton x={x} y={y} mag={mag} depth={depth} />
                </Stack>
            </Grid>

            <Grid xs={12} lg={6}>
                <Stack
                    alignItems={"center"}
                    justifyItems={"center"}
                    direction={"column"}
                >
                    <MapOfJapan></MapOfJapan>
                </Stack>
            </Grid>
        </Grid>
    );
};

export default English;
