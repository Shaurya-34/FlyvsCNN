export interface DroneConfig {
  speed: number; // constant forward speed (m/s)
  lateralDamping: number;
}

export interface DroneState {
  x: number; // lateral position
  z: number; // forward distance travelled
  lateralVel: number;
}

export function createDrone(): DroneState {
  return { x: 0, z: 0, lateralVel: 0 };
}

// steering in [-1, 1]
export function stepDrone(state: DroneState, steering: number, dt: number, config: DroneConfig): void {
  const steerAccel = 4; // engineering constant, tune later — not biological
  state.lateralVel += steering * steerAccel * dt;
  state.lateralVel *= Math.max(0, 1 - config.lateralDamping * dt);
  state.x += state.lateralVel * dt;
  state.z += config.speed * dt;
}
