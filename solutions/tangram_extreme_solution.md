# Tangram Cube Extreme solution

Solved with the ladder solver (learned_r7, beam 4096), then replayed on the reference simulator: the result is the solved state.

- Scramble (86 steps, 57 turns): `D F U' D2 L U2 D' F U2 D2' R U2 D F U3 R U3 D L U2 D3 R U2' D L U' D R U' D2 L U2 D2 R U' D3 F U' D2 R U3 D2' F U D' L U' D2' R D2' F U3 D2 R U3 D F`
- Solution: 80 moves as written below (84 steps, 80 turns)
- Per rung, in steps before merging across rungs: frame 6, large 46, small 40
- Notation: `U` and `D` turn the top and bottom corner layers 60° clockwise, as seen from that corner, and `'` means anticlockwise. A number repeats the turn, so `U3` is a half turn and `U2'` is 120° anticlockwise. `F`, `R` and `L` are 180° flips about three edges; each is its own inverse. This is the notation of the puzzle's simulator.
- Read left to right, top to bottom, 4 moves per line.

```
  1. F R F R
  2. F L R L
  3. U L R U'
  4. D F D2' F
  5. D2 U L U'
  6. L U' F D'
  7. F U2 L U
  8. L U' F D'
  9. F D2 U' R
 10. D' U' R F
 11. U F L U
 12. L F L F
 13. U' F L R
 14. U R F R
 15. F U' F R
 16. L D' L F
 17. L F D F
 18. L F R U
 19. R L R U'
 20. R U L U'
```
