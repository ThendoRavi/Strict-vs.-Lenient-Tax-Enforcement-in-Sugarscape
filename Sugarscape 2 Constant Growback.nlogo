extensions [csv]

globals [
  tax-revenue
  gini-index
  ;initial-population
  ;audit-percentage
  ;punishment-mode
  ;punishment-duration
  ;visualization-mode
  python-action-received
]

turtles-own [
  sugar
  metabolism
  vision
  vision-points
  strategy
  tax-due
  tax-paid
  compliance-history
  is-punished?
  punishment-timer
  python-action
  sugar-level
]

patches-own [
  psugar
  max-psugar
]

; Setup Procedures
to setup
  clear-all
  set tax-revenue 0
  set initial-population 200
  set audit-percentage 0.3
  set punishment-mode "strict"
  set punishment-duration 5
  set visualization-mode "none"
  set python-action-received false

  setup-patches
  create-turtles initial-population [ turtle-setup ]
  reset-ticks
  do-visualization
end

to turtle-setup
  set color green
  set shape "circle"
  move-to one-of patches with [not any? turtles-here]
  set sugar random-in-range 5 25
  set metabolism random-in-range 1 4
  set vision random-in-range 1 6
  set vision-points []
  set is-punished? false
  set punishment-timer 0
  set compliance-history []
  set python-action 0
  update-sugar-level

  ; Build vision points
  foreach (range 1 (vision + 1)) [ n ->
    set vision-points sentence vision-points (list (list 0 n) (list n 0) (list 0 (- n)) (list (- n) 0))
  ]
end

to setup-patches
  ; Create a simple sugar landscape
  ask patches [
    set max-psugar random 5 + 1
    set psugar max-psugar
    patch-recolor
  ]
end

; Runtime Loop - Modified for Python control
to go
  if not any? turtles [ stop ]

  ; Check if we've received actions from Python
  if python-action-received [
    update-punishment-status
    handle-tax
    audit-and-punish

    ask patches [
      patch-growback
      patch-recolor
    ]

    ask turtles [
      if not is-punished? [
        turtle-move
        turtle-eat
      ]
      if sugar <= 0 [ die ]
      update-sugar-level
    ]

    redistribute-tax
    do-visualization
    tick

    ; Reset flag for next cycle
    set python-action-received false
  ]
end

; Tax Logic
to handle-tax
  ask turtles [
    if is-punished? [ stop ]

    ; Calculate tax based on sugar
    set tax-due calculate-tax

    ; Use python-action to determine strategy
    if python-action = 0 [
      set strategy "pay"
      set tax-paid tax-due
    ]
    if python-action = 1 [
      set strategy "partial"
      set tax-paid tax-due * 0.5
    ]
    if python-action = 2 [
      set strategy "evade"
      set tax-paid 0
    ]

    ; Pay tax
    set sugar sugar - tax-paid
    set tax-revenue tax-revenue + tax-paid

    ; Record compliance
    if tax-paid = tax-due [ set compliance-history lput "full" compliance-history ]
    if tax-paid > 0 and tax-paid < tax-due [ set compliance-history lput "partial" compliance-history ]
    if tax-paid = 0 [ set compliance-history lput "none" compliance-history ]
  ]
end

to-report calculate-tax
  if sugar < 10 [ report sugar * 0.1 ]
  if sugar >= 10 and sugar < 30 [ report sugar * 0.2 ]
  if sugar >= 30 and sugar < 50 [ report sugar * 0.3 ]
  report sugar * 0.4
end

to audit-and-punish
  let candidates turtles with [ not is-punished? ]
  let audited-sample n-of (audit-percentage * count candidates) candidates
  ask audited-sample [
    if tax-paid < tax-due [
      if punishment-mode = "strict" [
        set is-punished? true
        set punishment-timer punishment-duration
        set sugar sugar - (tax-due - tax-paid) * 2
      ]
      if punishment-mode = "lenient" [
        set is-punished? true
        set punishment-timer punishment-duration
        set sugar sugar - (tax-due - tax-paid) * 0.5
      ]
    ]
  ]
end

to update-punishment-status
  ask turtles [
    if is-punished? [
      set punishment-timer punishment-timer - 1
      if punishment-timer <= 0 [
        set is-punished? false
        set punishment-timer 0
      ]
    ]
  ]
end

to redistribute-tax
  ; Simple redistribution: add sugar to random patches
  let redistribution-pool tax-revenue
  set tax-revenue 0

  while [redistribution-pool > 0] [
    ask one-of patches [
      set psugar psugar + 1
      set redistribution-pool redistribution-pool - 1
      patch-recolor
    ]
  ]
end

; Turtle Movement and Eating
to turtle-move
  let move-candidates (patch-set patch-here (patches at-points vision-points)) with [not any? turtles-here]
  let possible-winners move-candidates with-max [psugar]
  if any? possible-winners [
    move-to min-one-of possible-winners [distance myself]
  ]
end

to turtle-eat
  set sugar (sugar - metabolism + psugar)
  set psugar 0
end

; Patch Utilities
to patch-recolor
  set pcolor scale-color yellow psugar 0 10
end

to patch-growback
  set psugar min (list max-psugar (psugar + 1))
end

; sugar level helper
to update-sugar-level
  ifelse sugar < 0 [
    set sugar-level 0
  ] [
    set sugar-level min list 9 floor(sugar / 10)
  ]
end

; Python Integration Functions
to receive-actions [action-list]
  let id 0
  let list-length length action-list
  while [id < list-length and id < count turtles] [
    let current-action item id action-list
    ask turtle id [ set python-action current-action ]
    set id id + 1
  ]
  set python-action-received true
end

to-report report-states
  let result []
  ask turtles [
    let punished-status 0
    if is-punished? [ set punished-status 1 ]
    set result lput (list who sugar-level punished-status (length compliance-history)) result
  ]
  report result
end

to-report report-rewards
  report [sugar] of turtles
end

to set-params [audit mode duration]
  set audit-percentage audit
  set punishment-mode mode
  set punishment-duration duration
end

to-report get-population
  report count turtles
end

; Experiment and Data Handling
to update-all-plots
  set-current-plot "Tax Decisions"
  set-current-plot-pen "pay"
  plot count turtles with [strategy = "pay"]
  set-current-plot-pen "partial"
  plot count turtles with [strategy = "partial"]
  set-current-plot-pen "evade"
  plot count turtles with [strategy = "evade"]

  set-current-plot "Total Sugar"
  set-current-plot-pen "sugar"
  plot sum [sugar] of turtles

  set-current-plot "Wealth Distribution"
  clear-plot
  histogram [sugar] of turtles

  set-current-plot "Evasion Rate Over Time"
  set-current-plot-pen "evasion-rate"
  if count turtles > 0 [
    plot (count turtles with [strategy = "evade"]) / count turtles
  ]

  set gini-index calculate-gini [sugar] of turtles
  set-current-plot "Inequality Metrics"
  set-current-plot-pen "gini"
  plot gini-index
end

; Visualization
to do-visualization
  ask turtles [
    if visualization-mode = "none" [ set color blue ]
    if visualization-mode = "vision" [ set color scale-color red vision 1 6 ]
    if visualization-mode = "metabolism" [ set color scale-color green metabolism 1 4 ]
    if visualization-mode = "strategy" [
      if strategy = "pay" [ set color green ]
      if strategy = "partial" [ set color yellow ]
      if strategy = "evade" [ set color red ]
    ]
  ]
end

; Utilities
to-report random-in-range [low high]
  report low + random (high - low + 1)
end

to-report calculate-gini [ wealth-list ]
  if length wealth-list < 2 [ report 0 ]

  let sorted-wealth sort wealth-list
  let n length sorted-wealth
  let wealth-sum sum sorted-wealth
  let i 0
  let weighted-sum 0

  foreach sorted-wealth [ w ->
    set weighted-sum weighted-sum + (i + 1) * w
    set i i + 1
  ]

  let gini (2 * weighted-sum) / (n * wealth-sum) - (n + 1) / n
  report gini
end

; Export data for analysis
to export-data [filename]
  let headers ["who" "sugar" "metabolism" "vision" "strategy" "tax_due" "tax_paid" "punished" "compliance_history"]
  let data []

  ask turtles [
    let punished-status 0
    if is-punished? [ set punished-status 1 ]
    let row (list who sugar metabolism vision strategy tax-due tax-paid punished-status compliance-history)
    set data lput row data
  ]

  csv:to-file (word filename ".csv") (fput headers data)
end
@#$#@#$#@
GRAPHICS-WINDOW
295
10
659
375
-1
-1
7.12
1
10
1
1
1
0
1
1
1
0
49
0
49
1
1
1
ticks
30.0

BUTTON
10
55
90
95
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
100
55
190
95
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
200
55
290
95
go once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

CHOOSER
10
105
290
150
visualization
visualization
"no-visualization" "color-agents-by-vision" "color-agents-by-metabolism"
0

PLOT
670
10
890
165
Population
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plotxy ticks count turtles"

PLOT
900
10
1120
165
Wealth distribution
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 1 -16777216 true "" "set-histogram-num-bars 10\nset-plot-x-range 0 (max [sugar] of turtles + 1)\nset-plot-pen-interval (max [sugar] of turtles + 1) / 10\nhistogram [sugar] of turtles"

SLIDER
10
15
290
48
initial-population
initial-population
10
1000
100.0
10
1
NIL
HORIZONTAL

PLOT
670
175
890
330
Average vision
NIL
NIL
0.0
10.0
0.0
6.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plotxy ticks mean [vision] of turtles"

PLOT
900
175
1120
330
Average metabolism
NIL
NIL
0.0
10.0
0.0
5.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plotxy ticks mean [metabolism] of turtles"

MONITOR
20
160
115
209
population
count turtles
17
1
12

MONITOR
20
230
237
275
Count the numbe rof turtles who pay
count turtles with [strategy = \"pay\"]
17
1
11

PLOT
730
360
930
510
tax collected over time
ticks
tax-revenue
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot count turtles"

PLOT
10
295
210
445
Tax Decisions
ticks
turtles
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"pay" 1.0 0 -13840069 true "" ""
"partial" 1.0 0 -1184463 true "" ""
"evade" 1.0 0 -2674135 true "" ""

PLOT
940
355
1140
505
Total Sugar
ticks
sum
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot count turtles"
"sugar" 1.0 0 -7858858 true "" ""

SLIDER
355
390
527
423
audit-percentage
audit-percentage
0
1
0.3
0.05
1
NIL
HORIZONTAL

CHOOSER
555
385
693
430
punishment-mode
punishment-mode
"strict" "lenient"
0

SLIDER
370
445
542
478
punishment-duration
punishment-duration
5
50
5.0
5
1
NIL
HORIZONTAL

CHOOSER
220
435
358
480
visualization-mode
visualization-mode
"none" "vision" "metabolism"
0

PLOT
1135
175
1335
325
Evasion Rate Over Time
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot count turtles"
"evasion-rate" 1.0 0 -7500403 true "" ""

PLOT
1125
15
1325
165
Inequality Metrics
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot count turtles"
"gini" 1.0 0 -7500403 true "" ""

@#$#@#$#@
## WHAT IS IT?

This second model in the NetLogo Sugarscape suite implements Epstein & Axtell's Sugarscape Constant Growback model, as described in chapter 2 of their book Growing Artificial Societies: Social Science from the Bottom Up. It simulates a population with limited, spatially-distributed resources available. It differs from Sugarscape 1 Immediate Growback in that the growback of sugar is gradual rather than instantaneous.

## HOW IT WORKS

Each patch contains some sugar, the maximum amount of which is predetermined. At each tick, each patch regains one unit of sugar, until it reaches the maximum amount. The amount of sugar a patch currently contains is indicated by its color; the darker the yellow, the more sugar.

At setup, agents are placed at random within the world. Each agent can only see a certain distance horizontally and vertically. At each tick, each agent will move to the nearest unoccupied location within their vision range with the most sugar, and collect all the sugar there.  If its current location has as much or more sugar than any unoccupied location it can see, it will stay put.

Agents also use (and thus lose) a certain amount of sugar each tick, based on their metabolism rates. If an agent runs out of sugar, it dies.

## HOW TO USE IT

Set the INITIAL-POPULATION slider before pressing SETUP. This determines the number of agents in the world.

Press SETUP to populate the world with agents and import the sugar map data. GO will run the simulation continuously, while GO ONCE will run one tick.

The VISUALIZATION chooser gives different visualization options and may be changed while the GO button is pressed. When NO-VISUALIZATION is selected all the agents will be red. When COLOR-AGENTS-BY-VISION is selected the agents with the longest vision will be darkest and, similarly, when COLOR-AGENTS-BY-METABOLISM is selected the agents with the lowest metabolism will be darkest.

The four plots show the world population over time, the distribution of sugar among the agents, the mean vision of all surviving agents over time, and the mean metabolism of all surviving agents over time.

## THINGS TO NOTICE

The world has a carrying capacity, which is lower than the initial population of the world. Agents who are born in sugarless places or who consume more sugar than the land cannot be supported by the world, and die. Other agents die from competition - although some places in the world have enough sugar to support them, the sugar supply is limited and other agents may reach and consume it first.

As the population stabilizes, the average vision increases while the average metabolism decreases. Agents with lower vision cannot find the better sugar patches, while agents with high metabolism cannot support themselves. The death of these agents causes the attribute averages to change.

## THINGS TO TRY

How dependent is the carrying capacity on the initial population size?  Is there a direct relationship?

## EXTENDING THE MODEL

How does changing the amount or rate of sugar growback affect the behavior of the model?

## NETLOGO FEATURES

All of the Sugarscape models create the world by using `file-read` to import data from an external file, `sugar-map.txt`. This file defines both the initial and the maximum sugar value for each patch in the world.

Since agents cannot see diagonally we cannot use `in-radius` to find the patches in the agents' vision.  Instead, we use `at-points`.

## RELATED MODELS

Other models in the NetLogo Sugarscape suite include:

* Sugarscape 1 Immediate Growback
* Sugarscape 3 Wealth Distribution

## CREDITS AND REFERENCES

Epstein, J. and Axtell, R. (1996). Growing Artificial Societies: Social Science from the Bottom Up.  Washington, D.C.: Brookings Institution Press.

## HOW TO CITE

If you mention this model or the NetLogo software in a publication, we ask that you include the citations below.

For the model itself:

* Li, J. and Wilensky, U. (2009).  NetLogo Sugarscape 2 Constant Growback model.  http://ccl.northwestern.edu/netlogo/models/Sugarscape2ConstantGrowback.  Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

Please cite the NetLogo software as:

* Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

## COPYRIGHT AND LICENSE

Copyright 2009 Uri Wilensky.

![CC BY-NC-SA 3.0](http://ccl.northwestern.edu/images/creativecommons/byncsa.png)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.  To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 559 Nathan Abbott Way, Stanford, California 94305, USA.

Commercial licenses are also available. To inquire about commercial licenses, please contact Uri Wilensky at uri@northwestern.edu.

<!-- 2009 Cite: Li, J. -->
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
