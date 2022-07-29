import {
    Grid,
    Slider,
    Stack,
    styled,
    Typography,
    useTheme,
} from "@mui/material";
import { Dispatch, FC, SetStateAction, useState } from "react";

const Container = styled(Stack)(({ theme }) => ({
    margin: theme.spacing(2),
    width: "80%",
}));

type Props = {
    title: string;
    min: number;
    max: number;
    step: number;
    value: number;
    setValue: Dispatch<SetStateAction<number>>;
};

const InputBar: FC<Props> = ({ title, min, max, step, value, setValue }) => {
    const theme = useTheme();
    return (
        <Container gap={theme.spacing(3)}>
            <Typography variant="h3">
                {title} : {value}
            </Typography>
            <Slider
                min={min}
                max={max}
                step={step}
                value={value}
                defaultValue={(max + min) / 2}
                aria-label="Default"
                valueLabelDisplay="auto"
                color="secondary"
                onChange={(_event: Event, newValue: number | number[]) => {
                    setValue(newValue as number);
                }}
            />
        </Container>
    );
};

export default InputBar;
