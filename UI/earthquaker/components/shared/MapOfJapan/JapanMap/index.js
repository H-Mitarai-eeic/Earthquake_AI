import React from "react";
import PropTypes from "prop-types";
import { geoMercator, geoPath } from "d3-geo";
import { feature } from "topojson-client";
import { SCALE } from "../../global";

const width = 512 * SCALE;
const height = 576 * SCALE;

class JapanMap extends React.Component {

    static propTypes = {
        width: PropTypes.number,
        height: PropTypes.number,
        defaultColor: PropTypes.string,
        strokeColor: PropTypes.string,
        prefectures: PropTypes.object,
    };

    static defaultProps = {
        width: width,
        height: height,
        defaultColor: "#888888",
        strokeColor: "#333333",
        prefectures: {},
    };

    constructor(props) {
        super(props);
        this.state = {
            prefectures: [],
        };
    }

    projection() {
        return geoMercator().scale(1595 * SCALE).center([137.2, 38.38]).translate([width / 2, height / 2]);
    }

    componentDidMount() {
        fetch("/static/japan.topojson").then((response) => {
            if (response.status !== 200) {
                console.error(`There was a problem: ${response.status}`);
                return;
            }
            response.json().then((worldData) => {
                this.setState({
                    prefectures: feature(worldData, worldData.objects.japan)
                        .features,
                });
            });
        });
    }

    render() {
        return (
            <svg width={width} height={height}>
                <g className="prefectures">
                    {this.state.prefectures.map((d, i) => {
                        const prefecture = d.properties.nam_ja;
                        const color =
                            (this.props.prefectures[prefecture] &&
                                this.props.prefectures[prefecture].color) ||
                            this.props.defaultColor;
                        return (
                            <path
                                key={`path-${i}`}
                                d={geoPath().projection(this.projection())(d)}
                                className="prefecture"
                                fill={color}
                                stroke={this.props.strokeColor}
                                strokeWidth={0.5}
                            />
                        );
                    })}
                </g>
            </svg>
        );
    }
}

export default JapanMap;
