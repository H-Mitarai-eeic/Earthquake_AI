import React, { ReactNode } from "react";
import Head from "next/head";
import { Box, useTheme } from "@mui/material";
import theme from "../styles/theme";

type Props = {
    children: ReactNode;
    title?: string;
};
const Layout = ({ children, title = "This is the default title" }: Props) => (
    <Box margin={theme.spacing(12)}>
        <Head>
            <title>{title}</title>
            <meta charSet="utf-8" />
            <meta
                name="viewport"
                content="initial-scale=1.0, width=device-width"
            />
            <link
                rel="preconnect"
                href="https://fonts.gstatic.com"
                crossOrigin="true"
            />
            <link
                href="https://fonts.googleapis.com/css2?family=Fredoka&family=Kosugi+Maru&family=Lora&family=Shippori+Mincho&family=Ubuntu:wght@300&display=swap"
                rel="stylesheet"
            />
        </Head>
        {children}
    </Box>
);

export default Layout;
