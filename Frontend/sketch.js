let players = [];
let selectedPlayer = null;
let drawingMode = "formation"; // or "movement"
let isPlaying = false;
let frame = 0;
const BALL_PATH = [];
let pitchTexture;

// Add these lines at the top, after your let declarations:
let margin, pitchLeft, pitchTop, pitchWidth, pitchHeight, centerX, centerY;

function setup() {
  createCanvas(1000, 600);

  // Define pitch geometry globally so all functions use the same values
  margin = 20;
  pitchLeft = margin;
  pitchTop = margin;
  pitchWidth = width - 2 * margin;
  pitchHeight = height - 2 * margin;
  centerX = width / 2;
  centerY = height / 2;

  // Generate pitch texture once
  pitchTexture = createGraphics(width, height);
  pitchTexture.noStroke();
  for (let i = 0; i < 30000; i++) {
    let x = random(width);
    let y = random(height);
    let alpha = random(10, 30);
    pitchTexture.fill(34 + random(10), 139 + random(20), 34 + random(10), alpha);
    pitchTexture.ellipse(x, y, 1, 1);
  }

  // Add this block to create the striped pattern
  for (let i = 0; i < 10; i++) {
    let stripeColor = i % 2 === 0 ? color(30, 120, 30) : color(40, 140, 40);
    pitchTexture.fill(stripeColor);
    pitchTexture.noStroke();
    pitchTexture.rect(i * (width / 10), 0, width / 10, height);
  }

  // Initialize 22 players with fixed spacing and IDs
  const spacing = height / 22;
  for (let i = 0; i < 22; i++) {
    const x = i < 11 ? width * 0.25 : width * 0.75;
    const y = spacing * i + spacing / 2 + (i < 11 ? 100 : -200); // Blue +50, Red +100
    players.push({
      id: i + 1,
      pos: createVector(x, y),
      path: [],
    });
  }

  // Ball path: from left penalty spot to right penalty spot (normalized)
  const ballPathNorm = [
    [0.32214559386973174, 0.5037931034482759],
    [0.9711877394636015, 0.5126436781609196]
  ];

  // Convert normalized path to pitch coordinates
  BALL_PATH.length = 0; // Clear any previous path
  for (let i = 0; i < 100; i++) {
    let t = i / 99;
    let xNorm = lerp(ballPathNorm[0][0], ballPathNorm[1][0], t);
    let yNorm = lerp(ballPathNorm[0][1], ballPathNorm[1][1], t);
    let x = pitchLeft + xNorm * pitchWidth;
    let y = pitchTop + yNorm * pitchHeight;
    BALL_PATH.push(createVector(x, y));
  }
}

function draw() {
  drawPitch();

  // Draw ball path
  stroke(0);
  noFill();
  beginShape();
  for (let p of BALL_PATH) vertex(p.x, p.y);
  endShape();

  // Draw players
  for (let i = 0; i < players.length; i++) {
    let player = players[i];

    stroke(255);        // White outline
    strokeWeight(2);    // Outline thickness

    if (i < 11) {
      fill(0, 120, 255); // Blue for Home
    } else {
      fill(220, 50, 50); // Red for Away
    }
    ellipse(player.pos.x, player.pos.y, 25);

    noStroke(); // Remove stroke for text and other elements

    // Draw player number in thin font
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(13);
    textFont('monospace');
    textStyle(NORMAL);
    text(i % 11 + 1, player.pos.x, player.pos.y);

    // Draw role below player in black
    if (player.role) {
      fill(0);
      textSize(15);
      text(player.role, player.pos.x, player.pos.y + 23);
    }

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

  // Draw the ball
  let ballPos;
  if (isPlaying && frame < BALL_PATH.length) {
    ballPos = BALL_PATH[frame];
  } else {
    ballPos = BALL_PATH[0]; // Ball at start when not playing
  }
  stroke(255);        // White outline
  strokeWeight(2);
  fill(0);            // Black fill
  ellipse(ballPos.x, ballPos.y, 18); // Ball size
  noStroke();

  if (isPlaying) frame++;
}

function drawPitch() {
  // Margin for the pitch
  const margin = 20;
  const pitchLeft = margin;
  const pitchTop = margin;
  const pitchWidth = width - 2 * margin;
  const pitchHeight = height - 2 * margin;
  const centerX = width / 2;
  const centerY = height / 2;

  // Pitch base color
  background(34, 139, 34);

  // Overlay the grainy texture
  image(pitchTexture, 0, 0);

  // Pitch outline
  stroke(255);
  strokeWeight(3);
  noFill();
  rect(pitchLeft, pitchTop, pitchWidth, pitchHeight);

  // Halfway line
  line(centerX, pitchTop, centerX, pitchTop + pitchHeight);

  // Center circle
  ellipse(centerX, centerY, 120, 120);

  // Center spot
  fill(255);
  noStroke();
  ellipse(centerX, centerY, 8, 8);

  // Penalty areas
  stroke(255);
  noFill();
  // Left penalty box
  rect(pitchLeft, centerY - 110, 120, 220);
  // Right penalty box
  rect(width - margin - 120, centerY - 110, 120, 220);

  // 6-yard boxes
  rect(pitchLeft, centerY - 55, 40, 110);
  rect(width - margin - 40, centerY - 55, 40, 110);

  // Penalty spots
  fill(255);
  noStroke();
  ellipse(pitchLeft + 90, centerY, 8, 8); // Left
  ellipse(pitchLeft + pitchWidth - 90, centerY, 8, 8); // Right (symmetric)

  // Corner arcs
  arc(pitchLeft, pitchTop, 30, 30, 0, HALF_PI);
  arc(width - margin, pitchTop, 30, 30, HALF_PI, PI);
  arc(pitchLeft, height - margin, 30, 30, -HALF_PI, 0);
  arc(width - margin, height - margin, 30, 30, PI, PI + HALF_PI);
}

function mousePressed() {
  for (let player of players) {
    if (dist(mouseX, mouseY, player.pos.x, player.pos.y) < 10) {
      selectedPlayer = player;
      break;
    }
  }

  if (drawingMode === "movement" && selectedPlayer) {
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
  if (drawingMode === "formation") {
    selectedPlayer = null;
  }
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
  players.forEach(p => (p.path = []));
  frame = 0;
  isPlaying = false;
}

function assignRoles() {
  for (let i = 0; i < players.length; i++) {
    let team = i < 11 ? "Home (Blue)" : "Away (Red)";
    let role = prompt(`Enter role for ${team} Player ${i % 11 + 1} (e.g., LB, ST):`, players[i].role || "");
    players[i].role = role || "";
  }
}
