import { red, green, blue, grey } from "@mui/material/colors";
import { createTheme, responsiveFontSizes } from "@mui/material/styles";

const spacing = 6;
const fontFamily = [
    "Fredoka",
    "Fredoka sans-serif",
    "Kosugi Maru",
    "sans-serif",
    "Lora",
    "serif",
    "Shippori Mincho",
    "serif",
    "Ubuntu",
    "sans-serif",
];
const theme = createTheme({
    palette: {
        primary: {
            main: "#212121",
            light: "#484848",
            dark: "#000000",
        },
        secondary: {
            main: "#BEF67A", // TODO: Set color
        },
        common: {
            black: "#000000", // TODO: Set color
            white: "#FFFFFF",
        },
        info: {
            main: blue[500], // TODO: Set color
        },
        success: {
            main: green[500], // TODO: Set color
        },
        warning: {
            main: "#FFDB7B",
        },
        error: {
            main: red[500], // TODO: Set color
        },
        tonalOffset: 0.2,
        background: {
            default: "#FFFFFF",
            paper: "#FFFFFF",
        },
    },
    spacing: spacing,
    typography: {
        fontFamily: fontFamily.join(","),
        fontSize: 12,
    },
    components: {
        MuiTooltip: {
            styleOverrides: {
                popper: {
                    zIndex: 100000,
                },
                tooltip: {
                    backgroundColor: grey[100],
                },
            },
        },
    },
});

export const truncate = (width: number | string) => ({
    width: typeof width === "string" ? width : `${width}px`,
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
});

// export const theme = responsiveFontSizes(_theme);
export default responsiveFontSizes(theme);
