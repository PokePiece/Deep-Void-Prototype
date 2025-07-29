import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { writeFileSync } from 'fs';

const scene = new THREE.Scene();

const geometry = new THREE.BoxGeometry(10, 10, 10);
const material = new THREE.MeshStandardMaterial({ color: 0x808080 });
const staticMesh = new THREE.Mesh(geometry, material);
scene.add(staticMesh);

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(10, 10, 10);
scene.add(light);

const exporter = new GLTFExporter();

exporter.parse(
  scene,
  function (gltf) {
    writeFileSync('./static_scene.gltf', JSON.stringify(gltf));
    console.log('Static scene exported');
  },
  { binary: false }
);
