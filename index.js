const cv = require('opencv4nodejs');
const path = require('path');
const express = require('express');
const app = express();
const server = require('http').Server(app);
const io = require('socket.io')(server);
//Import PythonShell module. 
const {PythonShell} =require('python-shell'); 

const FPS = 10;

//const wCap = new cv.VideoCapture(0);

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

const video = document.querySelector('video');
const canvas = document.querySelector('canvas');
canvas.width = 480;
canvas.height = 360;
const button = document.querySelector('button');

button.onclick = function() {
    /* set the canvas to the dimensions of the video feed */
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    /* make the snapshot */
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  };

navigator.mediaDevices.getUserMedia( {audio: false, video: true })
.then(stream => video.srcObject = stream)
.catch(error => console.error(error)); 

// async function getMedia(constraints) {
//     let stream = null;
  
//     try {
//       stream = await navigator.mediaDevices.getUserMedia(constraints);
//       /* use the stream */

//     } catch(err) {
//       /* handle the error */
//     }
//   }
  

/*setInterval(() => {
    //const frame = wCap.read();
    //const image = cv.imencode('.jpg', frame).toString('base64');

    // var childProcess = require("child_process");
    // var oldSpawn = childProcess.spawn;
    // function mySpawn() {
    //     console.log('spawn called');
    //     console.log(arguments);
    //     var result = oldSpawn.apply(this, arguments);
    //     return result;
    // }
    // childProcess.spawn = mySpawn;


    let options = { 
        mode: 'text', 
        pythonOptions: ['-u'], // get print results in real-time 
        args: [image] //An argument which can be accessed in the script using sys.argv[1] 
    }; 
      
  
    PythonShell.run('card_detector.py', options, function (err, result){ 
          if (err) throw err; 
          // result is an array consisting of messages collected  
          //during execution of script. 
          console.log('result: ', result.toString()); 
          io.emit(result.toString()) 
    }); 

    // const spawn = require('child_process').spawn;
    // const pythonProcess = spawn('python',['hello.py']);
    // pythonProcess.stdout.on('data', (data) => {
    //     // send image url
    //     io.emit('data', data);
    // });    
}, 1000 / FPS)*/

server.listen(3000);

