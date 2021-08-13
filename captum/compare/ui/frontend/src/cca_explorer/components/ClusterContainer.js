import React, { useState, useRef, useEffect } from "react";
import { makeRequest } from "../services/api"
import { SamplesWindow } from "./Samples"

import "./ClusterContainer.css";

function ClusterContainer({ layerName, modelName }) {
    const [imgData, setImgData] = useState("");
    const [ startPoint, setStartPoint ] = useState([0, 0]);
    const [ endPoint, setEndPoint ] = useState([0, 0]);
    const [ imagesInCluster, setImagesInCluster ] = useState([0,1,2]);

    const [ startRect, setStartRect ] = useState(false);

    const canvasRef = useRef(null)

    useEffect(() => {
        async function getData() {
            const img_data = await makeRequest(`/clustered-gram/${modelName}/${layerName}`);
            setImgData(`data:image/png;base64, ${img_data}`)
        }
        getData();

    }, [layerName, modelName]);



    useEffect(() => {
        const imgs_in_cluster = async () => {
            const firstIdx = parseInt((startPoint[0]/630)*5000)
            const secondIdx = parseInt((endPoint[0]/630)*5000)
            const data = await makeRequest(`/cluster-images/${modelName}/${layerName}/${firstIdx}/${secondIdx}`)
            setImagesInCluster(data.data)
        }
        imgs_in_cluster();
    }, [endPoint, layerName, modelName])

    function getCursorPosition(canvas, event) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        return { x, y };
    }

    const onMouseDown = (e) => {
        const canvas = canvasRef.current
        const pos = getCursorPosition(canvas, e)
        setStartPoint([pos.x, -pos.x])
        setStartRect(true);
    }

    const onMouseUp = async (e) => {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d');
        const pos = getCursorPosition(canvas, e);
        const mousex = pos.x;
        setEndPoint([mousex, -mousex]);

        var width = mousex-startPoint[0];
        const yPos = canvas.height - ((startPoint[0])*(720/630))
        ctx.rect(startPoint[0],yPos,width, -width*(720/630));

        ctx.strokeStyle = 'white';
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 3;
        ctx.stroke();

        setStartRect(false);
    }

    const onMove = (e) => {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d');
        const pos = getCursorPosition(canvas, e);
        const mousex = pos.x;
        if(startRect) {
            ctx.clearRect(0,0,canvas.width,canvas.height); //clear canvas
            ctx.beginPath();
            var width = mousex-startPoint[0];
            const yPos = canvas.height - ((startPoint[0])*(720/630))
            ctx.rect(startPoint[0],yPos,width, -width*(720/630));
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.stroke();
        }
    }

    return (
        <>
        <div className="cluster-container">
            {/* <LazyImage srcProvider={img_url} className="cluster-img" /> */}
            <img className="cluster-img" src={imgData} width={700} height={800} alt="Cluster"/>
            <canvas
                ref={canvasRef}
                className="rect-canvas"
                width={630}
                height={720}
                onMouseMove={onMove}
                onMouseDown={onMouseDown}
                onMouseUp={onMouseUp}
                style={{
                    marginTop: "80px"
                }}
            />
        </div>
        <div>
        {parseInt((startPoint[0]/630)*5000)} {parseInt((endPoint[0]/630)*5000)}
        <SamplesWindow sampleIds={imagesInCluster} columns={6}/>
        </div>
        </>
    )
}

export default ClusterContainer
