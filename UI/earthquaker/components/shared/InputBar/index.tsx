import {
    Grid,
    Slider,
    Stack,
    styled,
    Typography,
    useTheme,
} from "@mui/material";
import { Dispatch, FC, SetStateAction, useState } from "react";
import { DEFAULT_DATA } from "../global";

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
    setData: Dispatch<SetStateAction<number[]>>;
    isLoading: boolean;
    unit?: string;
};

const InputBar: FC<Props> = ({
    title,
    min,
    max,
    step,
    value,
    setValue,
    setData,
    isLoading,
    unit,
}) => {
    const theme = useTheme();
    return (
        <Container gap={theme.spacing(3)}>
            <Typography variant="h3">
                {title} : {value} {unit}
            </Typography>
            <Slider
                disabled={isLoading}
                min={min}
                max={max}
                step={step}
                value={value}
                defaultValue={(max + min) / 2}
                aria-label="Default"
                valueLabelDisplay="auto"
                color="secondary"
                onChange={(_event: Event, newValue: number | number[]) => {
                    setData(DEFAULT_DATA);
                    setValue(newValue as number);
                }}
            />
        </Container>
    );
};

export default InputBar;
