let players = [];
let selectedPlayer = null;
let drawingMode = "formation"; // or "movement"
let isPlaying = false;
let frame = 0;
const BALL_PATH = [];

function setup() {
  createCanvas(1000, 600);
  // Initialize 22 players
  for (let i = 0; i < 22; i++) {
    players.push({
      id: i,
      pos: createVector(random(width), random(height)),
      path: [],
    });
  }
  // Create fixed ball path
  for (let i = 0; i < 100; i++) {
    let x = lerp(0.322 * width, 0.971 * width, i / 99);
    let y = lerp(0.504 * height, 0.513 * height, i / 99);
    BALL_PATH.push(createVector(x, y));
  }
}

function draw() {
  background(30);

  stroke(255);
  noFill();
  beginShape();
  for (let p of BALL_PATH) vertex(p.x, p.y);
  endShape();

  for (let player of players) {
    fill(255, 100, 100);
    ellipse(player.pos.x, player.pos.y, 15);

    if (drawingMode === "movement" && player.path.length > 0) {
      stroke(0, 255, 0);
      noFill();
      beginShape();
      for (let pt of player.path) vertex(pt.x, pt.y);
      endShape();

      if (isPlaying && frame < 100) {
        fill(0, 255, 0);
        let anim = player.path[Math.floor((frame / 100) * player.path.length)];
        ellipse(anim.x, anim.y, 12);
      }
    }
  }

  if (isPlaying) frame++;
}

function mousePressed() {
  if (drawingMode === "formation") {
    for (let player of players) {
      if (dist(mouseX, mouseY, player.pos.x, player.pos.y) < 10) {
        selectedPlayer = player;
        break;
      }
    }
  } else if (drawingMode === "movement" && selectedPlayer) {
    selectedPlayer.path.push(createVector(mouseX, mouseY));
  }
}

function mouseDragged() {
  if (drawingMode === "formation" && selectedPlayer) {
    selectedPlayer.pos = createVector(mouseX, mouseY);
  } else if (drawingMode === "movement" && selectedPlayer) {
    selectedPlayer.path.push(createVector(mouseX, mouseY));
  }
}

function mouseReleased() {
  if (drawingMode === "formation") selectedPlayer = null;
}

function toggleMode() {
  drawingMode = drawingMode === "formation" ? "movement" : "formation";
}

function deselectPlayer() {
  selectedPlayer = null;
}

function undoLastPoint() {
  if (selectedPlayer && selectedPlayer.path.length > 0) {
    selectedPlayer.path.pop();
  }
}

function saveFormation() {
  let payload = players.map(p => ({
    player_id: p.id,
    is_ball: false,
    path: p.path.map(pt => [pt.x / width, pt.y / height]),
  }));

  fetch("https://your-backend-url/save-formation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(res => res.json())
    .then(data => alert("Saved successfully!"))
    .catch(err => alert("Error saving formation"));
}

function submitForPrediction() {
  let payload = players.map(p => ({
    player_id: p.id,
    is_ball: false,
    path: p.path.map(pt => [pt.x / width, pt.y / height]),
  }));

  fetch("https://your-backend-url/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(res => res.json())
    .then(data => {
      // update paths based on prediction
      for (let i = 0; i < players.length; i++) {
        players[i].path = data[i].map(pt => createVector(pt[0] * width, pt[1] * height));
      }
      alert("Prediction loaded");
    })
    .catch(err => alert("Error predicting"));
}

function playAnimation() {
  isPlaying = true;
  frame = 0;
}

function pauseAnimation() {
  isPlaying = false;
}

function resetAll() {
  players.forEach(p => p.path = []);
  frame = 0;
  isPlaying = false;
}
