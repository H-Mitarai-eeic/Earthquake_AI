import {
    Box,
    Button,
    Grid,
    Modal,
    Stack,
    styled,
    Typography,
    useTheme,
} from "@mui/material";
import { Dispatch, FC, SetStateAction, useState } from "react";
import {
    BITSIZE,
    DEFAULT_DATA,
    Lang,
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

type Props = {
    language: Lang;
    modalOpen: boolean;
    setModalOpen: Dispatch<SetStateAction<boolean>>;
};

const Earthquaker: FC<Props> = ({ language, modalOpen, setModalOpen }) => {
    const theme = useTheme();
    const [pos, setPos] = useState([32, 32]);
    const [mag, setMag] = useState(7.5);
    const [depth, setDepth] = useState(350);
    const [isLoading, setIsLoading] = useState(false);
    const [data, setData] = useState(DEFAULT_DATA);
    const isEnglish = language == "English";

    return (
        <>
            <Grid container>
                <Grid xs={12} lg={6}>
                    <Stack
                        alignItems={"center"}
                        justifyItems={"center"}
                        gap={1}
                    >
                        <Typography
                            variant="h1"
                            textAlign={"left"}
                            marginBottom={2}
                        >
                            Earthquaker
                        </Typography>
                        <InputBar
                            title={isEnglish ? "magnitude" : "マグニチュード"}
                            min={5}
                            max={10}
                            step={0.1}
                            value={mag}
                            setValue={setMag}
                            isLoading={isLoading}
                            setData={setData}
                        />
                        <InputBar
                            title={isEnglish ? "depth" : "震源の深さ"}
                            min={0}
                            max={700}
                            step={1}
                            value={depth}
                            setValue={setDepth}
                            isLoading={isLoading}
                            setData={setData}
                            unit={"km"}
                        />
                        <Stack direction={"row"} gap={5}>
                            {isEnglish ? (
                                <>
                                    <Typography variant="h4">
                                        latitude :&nbsp;
                                        {Math.round(
                                            LATITUDE_MAX -
                                                (pos[1] * LATITUDE_SPAN) /
                                                    BITSIZE
                                        )}
                                        °N
                                    </Typography>
                                    <Typography variant="h4">
                                        longtitude :&nbsp;
                                        {Math.round(
                                            LONGTITUDE_MIN +
                                                (pos[0] * LONGTITUDE_SPAN) /
                                                    BITSIZE
                                        )}
                                        °E
                                    </Typography>
                                </>
                            ) : (
                                <>
                                    <Typography variant="h4">
                                        緯度 :&nbsp;北緯&nbsp;
                                        {Math.round(
                                            LATITUDE_MAX -
                                                (pos[1] * LATITUDE_SPAN) /
                                                    BITSIZE
                                        )}
                                        &nbsp;度
                                    </Typography>
                                    <Typography variant="h4">
                                        経度 :&nbsp;東経&nbsp;
                                        {Math.round(
                                            LONGTITUDE_MIN +
                                                (pos[0] * LONGTITUDE_SPAN) /
                                                    BITSIZE
                                        )}
                                        &nbsp;度
                                    </Typography>
                                </>
                            )}
                        </Stack>
                        <RunButton
                            x={pos[0]}
                            y={pos[1]}
                            mag={mag}
                            depth={depth}
                            setData={setData}
                            isLoading={isLoading}
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

            <Modal
                open={modalOpen && !isLoading}
                sx={{
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    transform: "translateY(-50%) translateX(-50%)",
                    WebkitTransform: "translateY(-50%) translateX(-50%)",
                    width: "100%",
                    height: "100%",
                }}
                onClose={() => {
                    setModalOpen(false);
                }}
            >
                <Stack
                    direction="column"
                    justifyContent="space-evenly"
                    alignItems="center"
                    spacing={2}
                    height="100%"
                    padding={theme.spacing(40)}
                >
                    <Typography
                        variant="h2"
                        color={theme.palette.secondary.main}
                    >
                        Presets
                    </Typography>
                    <Box
                        onClick={() => {
                            setPos([
                                Math.round(
                                    ((139 - LONGTITUDE_MIN) * BITSIZE) /
                                        LONGTITUDE_SPAN +
                                        2
                                ),
                                Math.round(
                                    ((LATITUDE_MAX - 35) * BITSIZE) /
                                        LATITUDE_SPAN +
                                        1
                                ),
                            ]);
                            setDepth(25);
                            setMag(7.9);
                            setData(DEFAULT_DATA);
                        }}
                    >
                        <Typography variant="h4">関東大震災</Typography>
                    </Box>
                    <Box
                        onClick={() => {
                            setPos([
                                Math.round(
                                    ((142 - LONGTITUDE_MIN) * BITSIZE) /
                                        LONGTITUDE_SPAN
                                ),
                                Math.round(
                                    ((LATITUDE_MAX - 38) * BITSIZE) /
                                        LATITUDE_SPAN
                                ),
                            ]);
                            setDepth(24);
                            setMag(9);
                            setData(DEFAULT_DATA);
                        }}
                    >
                        <Typography variant="h4">東日本大地震</Typography>
                    </Box>
                    <Box
                        onClick={() => {
                            setPos([
                                Math.round(
                                    ((134 - LONGTITUDE_MIN) * BITSIZE) /
                                        LONGTITUDE_SPAN
                                ),
                                Math.round(
                                    ((LATITUDE_MAX - 33) * BITSIZE) /
                                        LATITUDE_SPAN
                                ),
                            ]);
                            setDepth(35);
                            setMag(9);
                            setData(DEFAULT_DATA);
                        }}
                    >
                        <Typography variant="h4">南海トラフ</Typography>
                    </Box>

                    <Button
                        variant="contained"
                        onClick={() => {
                            setModalOpen(false);
                        }}
                    >
                        <Typography
                            variant="h4"
                            color={theme.palette.secondary.main}
                        >
                            CLOSE
                        </Typography>
                    </Button>
                </Stack>
            </Modal>
        </>
    );
};

export default Earthquaker;
