Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 1 items.

=== ITEM id=a94288701fc4b6b4 ===
Gold sentences and facts:
Sentence 0: It gave the JackJumpers a game-high 19-point lead and just about put the result to bed after Sydney had lost Jaylin Galloway (shoulder) and Jonah Bolden (fouled out).
  Fact 0.0: It gave the JackJumpers a game-high 19-point lead.
  Fact 0.1: The lead just about put the result to bed.
  Fact 0.2: Sydney had lost Jaylin Galloway (shoulder).
  Fact 0.3: Sydney had lost Jonah Bolden, who fouled out.
  Fact 0.4: The lead came after Sydney had lost Jaylin Galloway and Jonah Bolden.
Sentence 1: 9:20 p.m. — Nathaniel Lowe makes the play to retire Carroll, and that will bring the Rangers back up to bat with a 10-0 lead.
  Fact 1.0: At 9:20 p.m., Nathaniel Lowe makes the play to retire Carroll.
  Fact 1.1: The play by Nathaniel Lowe will bring the Rangers back up to bat.
  Fact 1.2: The Rangers will be back up to bat with a 10-0 lead.

Retrieved items:
<item 1>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Kangaroos’ Alice O’Loughlin and Melbourne’s Maeve Chaplin battle for the ball. Credit: AFL Photos It was a hard-fought, low-scoring battle between the two
powerhouse sides that play a similar game style. Each set up well, are fast and like a good dose of contested marking. But Narrm controlled more of the game by
tightening around the contest and being cleaner going forward. They also defended extremely well by suffocating the opportunities North created and prevailed
5.7 (37) to 2.2 (14). While the Demons scored just one goal a quarter in the second half – Tyla Hanks in the third and Eden Zanker in the fourth – the Kangaroos
were goalless for three terms, having led by five points at the first change. Their opening goals were thanks to Niamh Martin and Tess Craven.

Then a crafty goal from Demons skipper Kate Hore to open the second term put her side up, and Melbourne never gave up the lead.
</item 1>

<item 2>
World Series Game 2: Arizona Diamondbacks v Texas Rangers – live (The Guardian, 2023-10-28T23:00:28+00:00)
29 Oct 2023 02.37 GMT Diamondbacks 7-1 Rangers, bottom 8th Andrew Saalfrank is pitching now for Arizona as Kelly gets a well-earned rest. Rivera stays in the
game, playing third base. Lowe is at the plate for the Rangers and he promptly grounds out to second.



29 Oct 2023 02.36 GMT Me earlier: “I’m going to guess this ends with a score like Diamondbacks 6-2 Rangers.” If this holds up, I want to claim partial credit.



29 Oct 2023 02.34 GMT Diamondbacks 7-1 Rangers, top 8th Moreno strikes out on three pitches to finally end the inning, but the damage is done and this one feels
all but out of reach for Texas. Still, the game must go on and we head to the bottom of the 8th inning.
</item 2>

<item 3>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
11:04 p.m. — Jung quickly grounds into an inning-ending double play. Now the Diamondbacks just need six runs in the bottom of the ninth to extend the game.

11:02 p.m. — Saalfrank hits Carter, a call that is upheld after Torey Lovullo asked for a review.

10:59 p.m. — After Garver strikes out, Andrew Saalfrank is coming in to face Evan Carter.

10:56 p.m. — Seager leads off the ninth with a ground-rule double.

10:54 p.m. — Thomas hits a bouncer that ends the inning.

10:52 p.m. 3-RUN HOME RUN — Snakes alive? Gurriel launches a 3-run home run to give the crowd something to cheer about. The Rangers' lead is cut to six.

Rangers 11, Diamondbacks 2

10:50 p.m.
</item 3>

<item 4>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Magpies ruck Sabrina Frederick, who starred with 39 hit-outs and used her height to bring the ball to ground at the advantage of her midfielders, said her side
had been practising for games like this.

“Even in the chaos, it’s quite comfortable because we train it. We train what scenarios to do in what moments, and even though it’s a bit crazy, it felt so
calm. We put players in place and we know what we’re doing,” said Frederick. “But close games are good ones, for everyone. We knew they were going to come out
like that [in the last quarter]. When they’re down, they’re just going to give us everything.” Collingwood’s Brianna Davey kicks her team forward against
Geelong. Credit: Getty Frederick admitted at times she “couldn’t see” and felt like she could be “blown away”.
</item 4>

<item 5>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“I can name the whole team. Everyone had their second in the last couple minutes.

“That was such a team effort, everyone had a moment there when they had to put their head over the footy and lay a tackle and not let it slip.” Swanson (25
disposals, eight clearances) stood out for the Eagles, while Bombers workhorse Maddy Prespakis (30 disposals, nine clearances) was typically prominent in the
midfield alongside Georgia Nanscawen (21, six). With two rounds left, West Coast have a chance to build their winning streak when they meet the bottom of the
table Western Bulldogs next round. Slipping one place to sixth, Essendon have must-win matches against Carlton and Gold Coast on the road to the finals. The
Bombers will be without Steph Cain, who was sidelined with a concussion after an elbow to the head from Swanson.
</item 5>

<item 6>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Chris Goulding has fired off a season-high seven three-pointers to steer NBL ladder leaders Melbourne United to a 96-84 win over the Illawarra Hawks.

Operating without injured guards Ian Clark (hamstring) and Matthew Dellavedova (concussion) on Sunday, Melbourne suffered a first-half scare when Luke Travers
was forced off after a knock to his left knee.

It came as visitors Illawarra surged as much as 10 points clear during a high-scoring first half at John Cain Arena.

Facebook Twitter Whatsapp Reddit Email Share

But United captain Goulding’s hot hand kept his team in the contest as he drained 21 first-half points, including five three-pointers.

Goulding finished with 33 points including seven triples – more than any other player in a game this season – as United’s tough defence in the second half
carried them home.
</item 6>

<item 7>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
McVeigh (18 points) and Deng (17) gave strong offensive contributions for Tasmania, who lost Marcus Lee for the final few minutes after the starting centre
fouled out.

Taipans import injured in win over Kings

A nasty back injury to first-year import sensation Patrick Miller has marred the Taipans’ otherwise impressive 87-80 win over Sydney at the Cairns Convention
Centre.

Miller, who entered Saturday night’s clash ranked second in the NBL in scoring and fourth in assists, had the hosts in control before an explosive drive to the
bucket ended with him crashing to the floor, 1:09 out from halftime.

The American’s head whip-lashed onto the floor before he got up gingerly and walked slowly to the bench holding his lower back, unable to continue.
</item 7>

<item 8>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
10:20 p.m. — Cody Bradford is into the game to pitch for Texas.

10:17 p.m. — Nelson still gets through the inning unscathed as he preserves what's left of the Diamondbacks bullpen for Game 5.

10:15 p.m. — The first baserunner allowed by Nelson is Josh Jung, who has a three-hit night for the Rangers.

10:09 p.m. — It's just not the Diamondbacks' day (if you couldn't already tell). Gurriel grounds into an inning-ending double play to end that threat.

10:07 p.m. — Walker and Pham put back-to-back singles on the board with one out.

10:01 p.m. — Nelson strikes out Seager, whose World Series MVP case might be hurt by the fact he's hitting under .200 in the series. Onto the bottom of the
sixth.

9:59 p.m.
</item 8>

<item 9>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.
</item 9>

<item 10>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
A tough break for the Diamondbacks.

10:49 p.m.: Bottom of 9th — Rangers closer José Leclerc is in to try and secure the save.

10:46 p.m.: Top of 9th — Nathaniel Lowe hit a comebacker at Andrew Saalfrank but the ball ricocheted right to Ketel Marte who pitched it to Christian Walker to
end the inning.

10:44 p.m.: Top of 9th — Jonah Heim struck out swinging and with two outs, that'll do it for Luis Frias. LHP Andrew Saalfrank will come in to face LHH Nathaniel
Lowe to try and close out the inning.

10:39 p.m.: Top of 9th — Mitch Garver flew out to center and the Diamondbacks almost caught Evan Carter slipping, nearly doubling up the rookie at first.
Arizona challenged to see if Carter touched second base on his way back but the challenge was unsuccessful.

10:35 p.m.
</item 10>

<item 11>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
15 Rangers at Astros 8:15 p.m. Fox TEX 2, HOU 0 (TEX, 1-0) Oct. 16 Rangers at Astros 4:37 p.m. Fox or F TEX 5, HOU 4 (TEX, 2-0) Oct. 18 Astros at Rangers 8:03
p.m. FS1 HOU 8, TEX 5 (TEX, 2-1) Oct. 19 Astros at Rangers 8:03 p.m. FS1 HOU 10, TEX 3 (TIE, 2-2) Oct. 20 Astros at Rangers* 7:07 p.m. FS1 HOU 5, TEX 4 (HOU,
3-2) Oct. 22 Rangers at Astros* 8:03 p.m. FS1 TEX 9, HOU 2 (TIED, 3-3) Oct.
</item 11>

<item 12>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Check out all of the key moments you might have missed.

World Series Game 4 final score

1 2 3 4 5 6 7 8 9 F Rangers 0 5 5 0 0 0 0 1 0 11 Diamondbacks 0 0 0 1 0 0 0 4 2 7

World Series results. highlights from Rangers vs. Diamondbacks Game 4

(All times Eastern)

Final: Rangers 11, Diamondbacks 7

11:24 p.m. — Walker pops out in foul territory, and the Rangers ride an offensive explosion way back in the early innings to a Game 4 win. Texas is one win away
from its first championship. The Rangers can win it all as early as tomorrow night at Chase Field.
</item 12>

<item 13>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 13>

<item 14>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
NBL heavyweights United only kicked clear from last season’s wooden spooners towards the end of the third quarter following a high-scoring first half at WIN
Entertainment Centre on Friday.

Again sloppy from the field, the Hawks missed nine consecutive shots to finish the third period as United (5-1) opened up a 14-point lead that proved
match-winning.

“We just have these patches against really good basketball teams that let us down,” said Hawks coach Jacob Jackomas.

“The shots weren’t disgusting today, just some little things right now. We said that before with the team is that we need to do all the little things right
against this basketball team and we didn’t.”

Big man Gary Clark (17 points) was one of few Hawks who got his eye in, with imports Tyler Harvey (19 points, six from 16 attempts) and Justin Robinson (nine
points, two from eight) again struggling.
</item 14>

<item 15>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Melbourne led by as many as 16 points midway through the fourth quarter on the back of a three ball from Luke Travers (14 points, 13 rebounds), having outscored
the JackJumpers by 78 points to 49 after the first five minutes.

The JackJumpers pulled off a remarkable comeback on Melbourne back in round two and got as close as four points in the final minute, but United held firm for a
fifth straight win to stay on top at 7-1.

Dean Vickerman was celebrating his 300th game coaching in the NBL, and the Melbourne boss was full of praise for his team.

“Tonight was a little bit like my career,” he said.

“We didn’t start great but once we found our rhythm and especially the way we came out of the half and had a great third quarter, and held on down a few players
at the end there. But we held on.”
</item 15>

<item 16>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
: Bottom of 5th — Jon Gray is cruising. Despite an impromptu appearance, the Rangers pitcher has sent down the first six batters he has faced. Lourdes Gurriel
Jr. and Alek Thomas both grounded out and Evan Longoria popped out to first in foul play.

9:19 p.m.: Top of 5th — Brandon Pfaadt survives a lead-off walk to Nathaniel Lowe unscathed. He struck out Josh Jung, then Leody Tavares lined out to third and
Marcus Semien grounded out to third to end the inning.

Fourth inning: Rangers 3, Diamondbacks 0

9:07 p.m.: Bottom of 4th — Jon Gray didn't need much time to settle in. He struck out Gabriel Moreno, Christian Walker grounded out to third and Tommy Pham flew
out to center for an easy 1-2-3 inning.

9:03 p.m.
</item 16>

<item 17>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Moreno scores to put Arizona on the board.

9:33 p.m. — Tommy Pham works a 3-1 count, but Heaney comes right back and strikes him out looking.

9:30 p.m. — Walker quickly lines one down the left field line for a double. The Diamondbacks have runners on second and third with nobody out.

9:29 p.m. — Moreno walks to open the bottom of the fourth.

9:27 p.m. — Nelson retires Carter and Jung, so it's a much-needed quick inning by Diamondbacks pitching. Now it's just up to Arizona's offense to put up 10 or
so runs.

9:24 p.m. — Ryne Nelson strikes out Garver to open the fourth.

9:20 p.m. — Nathaniel Lowe makes the play to retire Carroll, and that will bring the Rangers back up to bat with a 10-0 lead.

9:17 p.m.
</item 17>

<item 18>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 18>

<item 19>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
😱



📺: FOX pic.twitter.com/83qi1EGjO2 — FOX Sports: MLB (@MLBONFOX) November 1, 2023

Rangers 7, Diamondbacks 0

9:08 p.m. 2-RUN DOUBLE — Who needs Adolis Garcia? Jankowski comes through with a massive 2-run double and puts the Rangers up 7-0. Texas is all over the
Diamondbacks' bullpen.

The Rangers tack on two more runs 😳



📺: FOX pic.twitter.com/AWYICKRMiq — FOX Sports: MLB (@MLBONFOX) November 1, 2023

9:06 p.m. — Luis Frias strikes out Taveras and has a chance to wiggle out of this against Jankowski.

9:03 p.m. — Disaster. Christian Walker had a chance to turn two but couldn't hold onto the ball.
</item 19>

<item 20>
World Series Game 2: Arizona Diamondbacks v Texas Rangers – live (The Guardian, 2023-10-28T23:00:28+00:00)
29 Oct 2023 03.21 GMT Final thoughts Okay, we officially have a series! Assuming one has no rooting interest in either of the remaining teams, there’s no
greater start to a World Series than a split in the first two games. A 1-1 split is an ideal situation for the casual fan as it usually portends a longer
series. It is, of course, a less than ideal situation for the Rangers. Today, their weaknesses were exposed and they know their two-game home stand could have
gone worse: they needed plenty of late-inning drama just to pull off a come-from-behind win in Game 1. What happens next is anybody’s guess. The World Series
heads to Arizona for Monday’s Game 3 where Rangers’ Max Scherzer and the Diamondbacks’ Brandon Pfaadt are currently penciled in as the probable starters. Stay
tuned to the Guardian for further coverage of the 2023 World Series as it develops but this will wrap up today’s Game 2 liveblog.
</item 20>

<item 21>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
— Semien strikes out to end the inning for the Rangers.

10:33 p.m. — The game was briefly paused while the public address announcer asked fans to stop throwing paper airplanes onto the field. That's where this game
is right now.

Rangers 11, Diamondbacks 1

10:29 p.m. HOME RUN — For the first time since the third inning, the Rangers have a run. Jonah Heim blasts a home run to right field to extend Texas' lead back
to 10.

The Heim Hammer 💥



📺: FOX pic.twitter.com/G4pO4pz9jv — FOX Sports: MLB (@MLBONFOX) November 1, 2023

10:24 p.m. — Jankowski collides with Semien in foul territory but makes a nice catch for the final out of a quick inning for Bradford.

10:20 p.m. — Cody Bradford is into the game to pitch for Texas.
</item 21>

<item 22>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
AAP Pies hold on against comeback Cats

Sunday’s clash between Collingwood and Geelong at Victoria Park was less about who was the overall better side than it was decided by who could adapt best to
the blustery wind and rain. The answer: the Magpies. Collingwood survived a fourth quarter comeback from Geelong to finish with a six-point victory and stay
firmly in finals contention, whereas the Cats now face an uphill battle to finish in the eight in two weeks’ time. Jordyn Allen of the Magpies is tackled by
Geelong’s Claudia Gunjaca in the round-eight match at Victoria Park. Credit: Getty Geelong scored just one point in the first quarter and had just 1.3 on the
board at three-quarter-time, but kicked two goals to none in the final term to close in on Collingwood.
</item 22>

<item 23>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But the Sixers went on a 10-0 run straight after the American went off, then moved in front for the first time since the opening quarter.

Sunday Dech nailed a three-pointer to give what was then the 36ers’ biggest lead of the day (seven points).

They never again trailed, clinching the final quarter 27-14.

Advertisement

“Zylan’s a focal point for us in everything that we do, he’s also one of our leaders, one of our best competitors,” Maor said.

“To see him go down like this, I think deflated us.

“New team, new people, new leaders, these are the kinds of things that we need to learn to deal with.”

The 36ers, tipped as possible wooden spooners in the pre-season, have now won three of their past four games.
</item 23>

<item 24>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Advertisement

But their hard work was undone when Creek, who helped himself to 13 points for the term, drained an uncontested three on the three-quarter-time buzzer to push
Phoenix out to 77-65, before Williams and Browne continued on their merry way in the fourth.

“Disappointing – that’s what that was,” Sixers coach CJ Bruton said.

“Our effort left, our communication left, and everyone was a step too slow.”

The Phoenix rose to 3-3 with their second straight win, while Adelaide (1-5) remain anchored to the bottom of the ladder.

CJ claims Kings didn’t look after DJ

Adelaide coach CJ Bruton has accused Sydney of not looking after Dejan Vasiljevic, as the war of words between the 36ers and Kings over the sharpshooting guard
escalated.

Vasiljevic, a two-time championship winner with Sydney, had originally been released from his Kings contract to chase his NBA dream.
</item 24>

<item 25>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Comeback kings Richmond take the points in Cairns A late goal by Emelia Yassir has helped Richmond to their first win in four games, putting their AFLW finals
campaign back on track with an 11-point victory over Hawthorn. Tight tussle: Richmond’s Emelia Yassir is tackled. Credit: Getty Images Separated by one point at
half-time and down by nine points entering the last quarter, the Tigers pulled off a 6.5 (41) to 4.6 (30) victory to keep their finals hopes alive at Cazaly’s
Stadium in Cairns on Friday night.

Hawthorn were limited to just one point in the decisive final quarter. Midfield duo Monique Conti (35 disposals, seven clearances) and Grace Egan (28, 5) were
typically influential for Richmond, while Hawthorn were led by Emily Bates (18, four) and Jasmine Fleming (18, one). “Richmond were just absolute beasts,
especially their midfielders.
</item 25>

<item 26>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The JackJumpers, who trailed by seven at three-quarter time, kept alive their record of winning every fourth term this season, but it wasn’t enough.

Brisbane captain Sobey led from the front as defensive specialist Mitch Norton kept Tasmania’s import Crawford (16 points) relatively quiet until the final
period.

Tyrrell Harrison and Josh Bannan (10 points each) were important for the Bullets, while Chris Smith (11) and Isaac White (10) gave solid contributions off the
bench.

The gut-wrenching loss ended a horror round five for the JackJumpers, who lost at home to Melbourne United before going down to the Bullets.

Advertisement

It left them with a 4-4 record after a strong 4-2 start to the season.
</item 26>

<item 27>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But lapses in concentration threatened to cost the Hawks.

They invited Cairns back into the game by shooting only one of their last nine attempts from the field in the third quarter, having gone almost four minutes
without scoring from the field in the second.

After the break, a free-throw from Sam Waardenburg tied things up, one of nine occasions in the final quarter alone that scores were level.

It took Harvey’s moment of magic to split the sides late.

“I’m really happy for the guys,” Jackomas said.

Advertisement

“It is a bit of a relief, but we need to move on into being a basketball team now, not the team that’s relieved to win.

“I’m really confident in these guys.”

Cheatham injured as 36ers upset Breakers

The New Zealand Breakers face a nervous wait after losing import Zylan Cheatham to a leg injury in the 97-85 upset loss to the Adelaide 36ers.
</item 27>

<item 28>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“I had a couple of opportunities where I’m open and I shot the ball,” Crawford said. “Marcus (Lee) and Majok were setting great screens and in order for me to
get those shots, them guys have to get me open.

“They did a hell of a job in doing that and it was a team effort. The ball was hot, the ball was moving and everybody contributed today.”

Advertisement

Sydney only had themselves to blame for the loss as they went 10-of-37 from three-point range.

Jaylen Adams (18 points, 10 rebounds) and Jordan Hunter (16, five) couldn’t lift the Kings over the line.

“Sometimes you just miss shots,” Kings coach Mahmoud Abdelfattah said.

“We’ve been practising these for the last two-and-a-half months, we got some great looks and at the end of the day that’s the difference.
</item 28>

<item 29>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
They finished with a staggering 23.

The Breakers were left to rue wayward free-throw shooting (12 from 20 attempts), which meant they were never able to open up a double-digit lead.

Former Golden State Warriors forward Anthony Lamb (25 points) is acclimatising quickly to the NBL, though the Breakers recruit had too little help as he notched
nine fourth-quarter points.

Advertisement

© AAP
</item 29>

<item 30>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Gun import Jordon Crawford has exploded late to lead the Tasmania JackJumpers to a club-record score in a 105-95 NBL road win over the defending champion Sydney
Kings.

The American guard opened the final period at Qudos Bank Arena on Sunday with four consecutive three-pointers and scored all of his team’s points on a stunning
14-2 run.

It gave the JackJumpers a game-high 19-point lead and just about put the result to bed after Sydney had lost Jaylin Galloway (shoulder) and Jonah Bolden (fouled
out).

Facebook Twitter Whatsapp Reddit Email Share

The Kings never gave up but could not close the gap as Crawford finished with a game-high 32 points.

Fellow import Milton Doyle and Majok Deng each scored 17 for Tasmania, who recorded their second win over Sydney in as many attempts this season.

“I had a couple of opportunities where I’m open and I shot the ball,” Crawford said.
</item 30>

<item 31>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
: Top of 3rd — Corey Seager rips a two-run home run! The Rangers produce some two-out magic as Seager smokes an inside changeup 421 feet over the right field
wall to give Texas a 3-0 lead.

8:41 p.m.: Top of 3rd — Marcus Semien comes up clutch with a single to left center! The star 2B has been struggling this postseason but he delivered here,
driving in Nathaniel Lowe to give the Rangers a 1-0 lead.

8:39 p.m.: Top of 3rd — Brandon Pfaadt struck out Josh Jung swinging and got Leody Tavares to ground out to second but Nathaniel Lowe advanced to third. He is
90 feet away with two outs and Marcus Semien is coming to the plate.

8:36 p.m.: Top of 3rd — Nathaniel Lowe gets the Rangers going right away with a double to right center field.
</item 31>

<item 32>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
“Kudos to them, they deserved it.”

Advertisement

Phoenix hold off fast-finishing Breakers

South East Melbourne Phoenix have survived a massive late scare to clinch a tense 103-100 victory over the fast-finishing New Zealand Breakers at John Cain
Arena.

The Phoenix were eyeing off a percentage pile-on when they led by a whopping 26 points on Saturday before the bottom-placed, injury-hit Breakers produced a
brave rearguard to almost pinch a miracle.

The Phoenix’s big four – Alan Williams (21 points, 13 rebounds), Mitch Creek (20 points), Gary Browne (18 points, eight assists) and Will Cummings (16 points,
all in the first half) – led the way but the final stages got too close for comfort.
</item 32>

<item 33>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Mitch Creek (23 points) led the Phoenix’s charge, import guards Gary Browne (19 points, eight assists) and Will Cummings (18 points) did as they pleased, while
Alan Williams (15 points, 13 rebounds) was influential in his first game of the season following a knee injury.

“We were trading baskets until that second five minutes of the second quarter when the guys got multiple stops in a row and kept scoring freely,” Phoenix coach
Mike Kelly said.

“That was the difference right there, that five-minute patch.”

Advertisement

Import Trey Kell (21 points) top-scored for Adelaide, while Dejan Vasiljevic, signed by the 36ers from the Sydney Kings only 48 hours earlier, had 17 points on
debut with his new club.

Vasiljevic was prolific early, firing 10 attempts and scoring 10 points in the first quarter.
</item 33>

<item 34>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Electrifying American point guard Parker Jackson-Cartwright (33 points, eight assists) and Finn Delany (21 points) were instrumental in the 32-19 fourth-quarter
fightback from the Breakers, who were brave despite succumbing to their fourth successive defeat.

“We were one lucky bounce away from winning this game,” NZ coach Mody Maor said. “Our guys are doing everything they can and everything I ask of them everyday.
“They are working as hard as can be. Our fans should be proud of the level of effort our guys put in.”

Advertisement

They were without import power forward Zylan Cheatham, who suffered a broken foot in last Thursday’s loss to Adelaide, and Williams duly proved a beast in the
low block against the undersized Anthony Lamb.

Behind Williams’ dominance and Browne’s ball-handling and playmaking wizardry, the Phoenix pulled ahead 31-21 at quarter-time.
</item 34>

<item 35>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
“I don’t want to discredit Illawarra (but) during the game, I felt like we were better-equipped to close it out, but we didn’t and Illawarra did. It’s the one
that got away.”

Nine first-quarter points from Froling helped the Hawks to a nine-point lead at the first change.

Froling (16 points, 13 rebounds) had his double-double with seven minutes remaining in the third quarter.

But Cairns did a better job at targeting him later on, keeping the rangy centre scoreless in the second half.

“He had a great game,” Jackomas said.

Advertisement

Import Justin Robinson, who had struggled from the field to begin the season, hit a three that forced a timeout and restored a game-high 13-point lead in the
third quarter.

But lapses in concentration threatened to cost the Hawks.
</item 35>

<item 36>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
That'll end the inning. A quick momentum shift back to the Rangers after the leadoff Marte single.

8:21 p.m. — Another strikeout for Heaney, who gets Moreno. The Diamondbacks are making him work, but he's getting the job done early.

8:18 p.m. — Heaney bounces back with a strikeout of Corbin Carroll.

8:15 p.m. — Ketel Marte's hitting streak reaches 20 as he singles up the middle to lead off the bottom of the first.

20-game postseason hitting streak for Ketel Marte 👏



📺: FOX pic.twitter.com/GJoEF1GVKj — FOX Sports: MLB (@MLBONFOX) November 1, 2023

8:12 p.m. — Carter flies out, so Mantiply gets through a scoreless first inning.
</item 36>

<item 37>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 37>

<item 38>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But Goulding caught fire to lead United to a 52-51 at the main break.

Defence was key in the second half as both sides tightened up, with Melbourne holding the visitors to just 15 points in the final period.

Travers returned after half-time, finishing with six points and a game-high 11 rebounds.

Advertisement

Ariel Hukporti retained his spot in Melbourne’s starting five ahead of Jo Lual-Acuil Jr and impressed with seven points and 10 rebounds, including four on the
offensive end.

Bullets ride late luck to win over Tasmania

Nathan Sobey has peeled off a game-high 28 points as Brisbane survived a late Tasmania fightback to continue their NBL resurgence with a tense 90-87 win.
</item 38>

<item 39>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
9:17 p.m. — At least Ketel Marte showed up for the Diamondbacks tonight. He's 2-2 after a double off of Heaney.

9:16 p.m. — Perdomo grounds out up the middle for the second out of the inning.

9:15 p.m. — Emmanuel Rivera opens the bottom of the third with a lineout.

Rangers 10, Diamondbacks 0

9:09 p.m. 3-RUN HOME RUN — This is ugly, ugly, ugly for Arizona. Marcus Semien blasts a 3-run home run to put the Rangers up by double-digits, and an Arizona
bullpen that has been so great in this postseason is getting bludgeoned.

A 3-RUN HOMER FROM SEMIEN MAKES IT 10-0 @Rangers! 😱
</item 39>

<item 40>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
The result was Melbourne’s ninth consecutive home win over New Zealand and gave them a 6-1 record for the season.

The Breakers (1-2) led by six points midway through the third period but ran out of steam in a tough loss that ended a gruelling road trip, which included two
exhibition matches against NBA opposition in the United States.

Advertisement

Zylan Cheatham (30 points) and Parker Jackson-Cartwright (18) threatened to drive the Breakers to a rousing victory until the guard fouled out inside the last
four minutes.

Former NBA forward Anthony Lamb had 15 points and five rebounds on his NBL debut for New Zealand after signing as a replacement for injured import Justinian
Jessup.

But Breakers coach Mody Maor was in no mood to look for positives in his team’s performance.

“We have very high standards for how we want to play and this isn’t it,” Maor said.
</item 40>

<item 41>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Bridie Hipwell kicked the only goal of the third quarter to give Hawthorn a game-high nine-point lead. Loading Richmond rebounded in the last term, with
Courtney Jones ending Hawthorn’s three-goal run with her first touch of the night, cutting the margin to three points. Dogged by inaccuracy in front of goal,
Yassir reduced the deficit to a single point before finally breaking through on her third attempt. That put the Tigers ahead by six points with 11 minutes left,
before Caitlin Greiser sealed the win with a massive goal from 45 metres.
</item 41>

<item 42>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But we held on.”

NBA and NBL champion Ian Clark top-scored with 19 points, with rookie Kyle Bowen adding a career-best 17 as did Jo Lual-Acuil along with 10 rebounds.

Advertisement

Jack McVeigh led the way for Tasmania with 20 points and seven rebounds, with Jordon Crawford contributing 16 points and four assists, and Milton Doyle 12
points and five assists.

Tasmania coach Scott Roth was philosophical afterwards, having given up 18 offensive rebounds and lost the battle on the glass by 16.

“We got a little uncharacteristic with our shot selection, which was a credit to them,” he said.

“Then they beat us up on the glass. It’s just little things when you get in those kind of games that are going to make the separation points.”
</item 42>

<item 43>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“He’s going to see double-teams down there so once we get our spacing right and our cutting right to play off that, he’s going to be a weapon,” Vickerman said.

Projected NBA draft lottery pick AJ Johnson had his best Hawks game yet, but was again short on opportunities.

The teenager set the crowd alight with his first NBL three-pointer just after quarter time and sent a dime through traffic that led to a Gary Clark jam on
transition soon afterwards.

Advertisement

Creek swamps Sixers for Phoenix

A powerful second quarter propelled South East Melbourne Phoenix to a clinical 102-85 NBL victory against the Adelaide 36ers at the Adelaide Entertainment
Centre.

The Phoenix pummelled the Sixers 28-14 in the second term on Thursday night to put the game out of the home side’s reach.
</item 43>

<item 44>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
28 Diamondbacks at Rangers 8:03 p.m. Fox ARI 9, TEX 1 (Tied 1-1) Oct. 30 Rangers at Diamondbacks 8:03 p.m. Fox TEX 3, ARI 1 (TEX, 2-1) Oct. 31 Rangers at
Diamondbacks 8:03 p.m. Fox -- Nov. 1 Rangers at Diamondbacks* 8:03 p.m. Fox -- Nov. 3 Diamondbacks at Rangers* 8:03 p.m. Fox -- Nov. 4 Diamondbacks at Rangers*
8:03 p.m. Fox --

*If necessary
</item 44>

<item 45>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Cummings scored at will in the second stanza before Creek inspired a 16-0 third-quarter burst for the home side, whose advantage swelled to 80-54.

NZ reverted to a zone defence and centre Mangok Mathiang impacted at both ends as the Breakers went 17-2 either side of three-quarter-time to haul themselves
back into the contest.

Quicksilver Cartwright-Jackson poured in 10 points inside three minutes to reduce the margin to three points down the stretch before Williams’ hook and Browne’s
free throw iced it narrowly.

“It was frustrating to see the lead dwindle,” Phoenix coach Mike Kelly said. “But it’s a win. The guys are going in the right direction. “I’m not disappointed,
I’m stoked.”
</item 45>

<item 46>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
J’Noemi Anderson is tackled by a group of Dockers. Credit: AFL Photos The Dockers dominated the Saints from the start to win 6.9 (45) to 3.3 (21) at Fremantle
Oval. Having been on the verge of making their first finals appearance, St Kilda (4-4) could now miss the top eight even if they beat the fourth-placed Brisbane
Lions and the ninth-placed Carlton.

Dana East was everywhere for Fremantle, leading the game in disposals (19) and contested possessions (11), alongside Orlagh Lally (19 disposals, five
clearances). Jamiee Lambert was the shining light for the otherwise subdued Saints, collecting a game-high seven clearances and 13 contested possessions. Docker
Hayley Miller hunts a loose ball against St Kilda. Credit: Getty Fremantle flew out of the gates from the first siren, producing their highest-scoring quarter
of the season with 26 points in the opening term.
</item 46>

<item 47>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
Even though they out-hit the Rangers, Arizona's bats hardly put any pressure on Texas' pitching.

MORE: Watch 2023 World Series games live with Fubo (free trial)

If you missed any of the action from Game 3 where the Rangers took a 2-1 series lead over the Diamondbacks, The Sporting News had you covered with live updates
and highlights below.

Rangers vs. Diamondbacks final score

1 2 3 4 5 6 7 8 9 F Rangers 0 0 3 0 0 0 0 0 0 3 Diamondbacks 0 0 0 0 0 0 0 1 0 1

World Series result, highlights from Game 3

(All times are Eastern.)

Ninth inning: Rangers 3, Diamondbacks 1

10:56 p.m.
</item 47>

<item 48>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“When you’re missing a ton of shots you’ve got to win the offensive rebounding battle and they beat us on that.”

Tasmania burst out to a double-figure lead when they piled on 36 points during the opening period, and the margin blew out to 18 before halftime.

Doyle (12 points), Deng and Jack McVeigh (both 10) all did plenty of damage on the scoreboard in the first half for the visitors, who never trailed. Galloway’s
huge dunk sparked Sydney early in the third period and they cut the margin to three.

Advertisement

But he and Bolden soon exited and Crawford’s scintillating run gave the JackJumpers the upper hand. The Kings are hopeful Galloway will be fit for Saturday’s
trip to take on the Cairns Taipans.

“The doctors may have other plans, but he’s doing alright,” Abdelfattah said.
</item 48>

<item 49>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Rangers 11, Diamondbacks 2

10:50 p.m. SACRIFICE FLY — Pham works a long at-bat against Stratton, and it ends in a sac-fly to trim the lead to nine.

10:44 p.m. — Now Walker delivers Arizona's third consecutive single, forcing Bradford out of the game. Chris Stratton is coming in with the bases loaded and one
out.

10:41 p.m. — Moreno follows Carroll with a single of his own. You have to give credit to the Diamondbacks' fans. They're still cheering pretty loudly.

10:40 p.m. — Corbin Carroll has struggled in this series, but he comes up with a one-out single here in the eighth.

10:34 p.m. — Semien strikes out to end the inning for the Rangers.

10:33 p.m.
</item 49>

<item 50>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The JackJumpers opened up on fire, racing to a 17-4 lead inside the opening five minutes including three balls to Crawford and McVeigh, but United soon steadied
and it was the introduction of Lual-Acuil that inspired it.

Melbourne would close the quarter outscoring Tasmania 17 points to four before three-pointers midway through the second stanza from Bowen and Clark put them up
five.

Advertisement

United were still leading 43-42 at halftime, on the back of 11 extra rebounds and 10 second-chance points, before captain Chris Goulding opened his account at
the start of the second half from downtown.

Clark hit his second long ball of the game soon after to push Melbourne’s lead out to nine, his third stretched it to double-figures and Bowen landed two
surprise ones of his own.
</item 50>

<item 51>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Hukporti was United’s rock under the basket and a defensive board in the final minute sealed his third double-double from five games this season.

Advertisement

“I think I’ve found my role in the team now, it took me a while because I’m coming off the (achilles) injury and it was a year out,” Hukporti said.

That is vintage Tyler Harvey ????



Can he pull the Hawks back into this one?



Catch the last quarter live on ESPN via Kayo ???? pic.twitter.com/40zoswVyPh — NBL (@NBL) October 20, 2023

“I’ve got it now, I know what my role is and I think I did a great job today.”

Lual-Acuil had a scratchy start, missing his first two free-throws and subbing out to change out of an incorrect pair of shorts, but went on to register 12
points on a minutes restriction.
</item 51>

<item 52>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Shea Ili (17 points) and Tanner Krebs (14) also reached double figures for Melbourne.

The result extended United’s winning streak to six games, giving them an 8-1 record on top of the ladder.

Import forward Gary Clark (24 points) and Justin Robinson (20) tried to give Illawarra a lift but Tyler Harvey managed just 10 points, three days after his
clutch three-pointer sealed a thrilling win over Cairns.

Advertisement

In a sloppy start, United committed five turnovers in the first period and trailed at quarter-time for the first time this season.

Clark’s eight points on perfect shooting led Illawarra to a 29-24 advantage at that point.

His tally swelled to 16 by the main break and the Hawks’ lead reached double figures soon after United temporarily lost Travers to injury.

But Goulding caught fire to lead United to a 52-51 at the main break.
</item 52>

<item 53>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
😮🛑



📺: FOX pic.twitter.com/wXM9RbieFp — FOX Sports: MLB (@MLBONFOX) October 31, 2023

8:25 p.m.: Bottom of 2nd — Christian Walker smoked a first-pitch fastball off the wall to right center for a double to lead off the bottom of the second.

8:22 p.m.: Top of 2nd — Rangers star rookie Evan Carter led off the top of the second with a hard-hit single to right center, but Mitch Garver topped a grounder
to Brandon Pfaadt for a 1-4-3 double play. Jonah Heim flew out to center to end the inning. Pfaadt has only needed 18 pitches to get through his first six
batters.

First inning: Rangers 0, Diamondbacks 0

8:15 p.m.
</item 53>

<item 54>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
15 Rangers at Astros 8:15 p.m. Fox TEX 2, HOU 0 (TEX, 1-0) Oct. 16 Rangers at Astros 4:37 p.m. Fox or F TEX 5, HOU 4 (TEX, 2-0) Oct. 18 Astros at Rangers 8:03
p.m. FS1 HOU 8, TEX 5 (TEX, 2-1) Oct. 19 Astros at Rangers 8:03 p.m. FS1 HOU 10, TEX 3 (TIE, 2-2) Oct. 20 Astros at Rangers* 7:07 p.m. FS1 HOU 5, TEX 4 (HOU,
3-2) Oct. 22 Rangers at Astros* 8:03 p.m. FS1 TEX 9, HOU 2 (TIED, 3-3) Oct.
</item 54>

<item 55>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Vasiljevic, fronting the press as a 36er for the first time on Wednesday, fired back at the Kings, saying “they made me look like a villain, they made Adelaide
look like the villain”.

But on Thursday night, Bruton called for Sydney to “move on”.

Advertisement

“I’ve only got my perspective – you (Kings) had the chance to have him, you chose not to have him,” he said.

“He’s moved on. Move on.

“If you wanted to do something more, look after the kid.

“Talk to him, championship ceremony or whatever … but that’s on them (Sydney).

“That’s not on us or me.”

Vasiljevic scored 17 points in his Adelaide debut on Thursday, including 11 in a productive first quarter, but he couldn’t prevent the bottom-placed Sixers from
suffering a 102-85 loss to South East Melbourne Phoenix.
</item 55>

<item 56>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“Lack of focus, missing key players, making mistakes on covers.

“I don’t care where we flew from, what our schedule is, what happened the day before – I care zero. This isn’t the level of execution that we expect from
ourselves.”

Shea Ili (16 points, four assists) was important for Melbourne and Luke Travers finished with 16 points, including 15 in the first half.

Sixers, Phoenix continue to rise

Advertisement

The Adelaide 36ers have told each other some hard truths before responding in style to heap more pressure on the Perth Wildcats with their 89-78 victory.

In the other NBL match on Saturday, South East Melbourne Phoenix made it three straight wins, beating the Bullets 96-73 as Alan Williams led the way against a
side looking lost without the suspended Aron Baynes.
</item 56>

<item 57>
World Series Game 2: Arizona Diamondbacks v Texas Rangers – live (The Guardian, 2023-10-28T23:00:28+00:00)
29 Oct 2023 02.50 GMT Diamondbacks 7-1 Rangers, top 9th Thomas is up with two on and one out. He grounds out on two pitches, moving the runners into scoring
position.



29 Oct 2023 02.49 GMT Diamondbacks 7-1 Rangers, top 9th Gurriel hits a single to centerfield, Peterson advances on the play.



29 Oct 2023 02.48 GMT Diamondbacks 7-1 Rangers, top 9th Jace Peterson is getting a pinch-hitting opportunity here for Arizona. It’s kind of like a basketball
game now here where the coaches empty out the bench. Peterson works a full count before hitting into a fielder’s choice. Walker is erased on the play. So it’s
one down with Peterson on first.
</item 57>

<item 58>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 58>

<item 59>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Mid-season recruit Dejan Vasiljevic (24 points, five assists) took centre stage for the Sixers after Cheatham went off.

The two-time NBL champion sank his 200th three-pointer in the first half and hit three more in the fourth quarter as the Sixers broke away.

Advertisement

“He’s a tough SOB,” 36ers coach CJ Bruton said.

“When it comes down to it, from the juniors, to the seniors, to the pros, that’s who he is and he carries that. He’s got respect from everyone, not only our
team, but the league.”

Sixers big man Isaac Humphries, who starred in an upset defeat of Perth last game, had been instrumental earlier on.

The centre (17 points, 11 rebounds) used his size advantage to dominate the paint and help Adelaide remain in the game, despite giving up 10 first-half
turnovers as a team.
</item 59>

<item 60>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
Ninth inning: Rangers 3, Diamondbacks 1

10:56 p.m.: Bottom of 9th — And José Leclerc strikes out both Christian Walker and Tommy Pham swinging to end the game! Rangers win! That's nine straight road
victories for Texas.

José Leclerc seals the W in Game 3 of the #WorldSeries for the @Rangers! 👏



📺: FOX pic.twitter.com/mvGRsBnLh3 — FOX Sports: MLB (@MLBONFOX) October 31, 2023

10:51 p.m.: Bottom of 9th — Gabriel Moreno thought he had a lead-off walk but the home plate umpire saw it as a strike even though it was off the plate. Moreno
grounded out to third on the next pitch. A tough break for the Diamondbacks.

10:49 p.m.
</item 60>

<item 61>
World Series Game 2: Arizona Diamondbacks v Texas Rangers – live (The Guardian, 2023-10-28T23:00:28+00:00)
Thanks to everybody who followed along with us tonight. Ciao!



29 Oct 2023 03.12 GMT Merrill Kelly Despite the final score, this was a much closer game until the Rangers bullpen imploded late. Arizona’s big hero was
starting pitcher Merrill Kelly who put up a dazzling pitching performance: 7.0 IP, 3 H, 1 ER, 7 Ks. It even felt like he could have stayed in there longer if
necessary. Easily the player of the game.
</item 61>

<item 62>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.
</item 62>

<item 63>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Williams produced 21 points and six rebounds for South East Melbourne but Brisbane will still be without Baynes for next weekend’s double-header away to Perth
and at home to Tasmania, and he needs to find some answers.

“We’ve got to have solutions right now for two more games without Banger (Baynes) and that’s the thing that’s sitting right in the front of my head, but scoring
73 points isn’t enough to get it done in this league,” Schueller said.

“Then giving them 96 is way too many as well so we have a lot of stuff that we still have to work on.”

JLA return lifts United to win in Wollongong

Advertisement

Melbourne United have celebrated the long-awaited return of Jo Lual-Acuil but it was fellow centre Ariel Hukporti who starred in a 101-91 defeat of the
lllawarra Hawks.
</item 63>

<item 64>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
Travis Janowski will take his place in right field.

10:15 p.m.: Top of 8th — Luis Frias does his job, sending down the Rangers' top of the order with ease. Marcus Semien grounded out to second, Corey Seager
struck out swinging and Adolis Garcia flew out to center. Something to monitor: Garcia came up holding his left side after the swing that ended the inning.

Adolis Garcia is coming out of the game following an apparent injury. pic.twitter.com/m5LV1b4QEX — The Sporting News (@sportingnews) October 31, 2023

10:10 p.m.: Top of 8th — RHP Luis Frias is in for Kyle Nelson after one inning of work.

Seventh inning: Rangers 3, Diamondbacks 0

10:07 p.m.: Bottom of 7th — Josh Sborz battles through to get out of a jam!
</item 64>

<item 65>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
“My words during the week were that we’re just not playing good, that showed again tonight,” Perth coach John Rillie said.

“Our offence isn’t functioning very good right now. We’re doing a great job holding teams to scores where we could put enough points on the board to win, in
saying that we’re not doing that.

“That’s certainly something we have to address and resolve here pretty quickly.”

Advertisement

Brisbane, who were on a four-game losing streak, started with the game’s first seven points and were still leading 20-9 by the end of the first quarter.

While the Wildcats had their moments in the second quarter, Brisbane did well to lead 42-36 at the long break.

It was 17-year-old Rocco Zikarsky who stood tall in the paint in the third quarter with Harrison in foul trouble.
</item 65>

<item 66>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Geelong coach Daniel Lowther said the strong winds made it hard for his side to clear the ball out of their defensive 50-metre zone and to move the ball with
any certainty around the ground.

Loading “At one point the rain hit our box, it was flowing,” Lowther said. “Both teams had to fight in the conditions, but it didn’t help the game. “Our marking
contest wasn’t too flash, our ground ball representation was a bit hit and miss, so our defence were under a bit more pressure than they should have been, but
they fought back really well,” Lowther said. Collingwood next play a Sydney line-up including former Magpies star Chloe Molloy, while Geelong will take on
Richmond at Ikon Park.

Dockers spoil Saints’ finals hopes Fremantle recorded a comfortable 24-point win yesterday to bring an abrupt end to St Kilda’s four-game streak and dash their
AFLW finals hopes. J’Noemi Anderson is tackled by a group of Dockers.
</item 66>

<item 67>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
A third straight defeat for the Wildcats has them struggling at 2-4 this season.

Coach John Rillie is focused know they aren’t playing well and the improvement needs to start offensively.

“We’re not playing good basketball at this point in time,” Rillie said.

“An area we need to improve on is our offence. That’s where we have to find some type of rhythm and we’ve yet to do that in these last couple of games.

Advertisement

“The game’s played at both ends and you need to find a rhythm to generate some natural enthusiasm, and we haven’t been able to do that for good stretches.”

Brisbane coach Justin Schueller always knew losing his superstar centre Baynes with a five-game suspension was going to be challenging, but that was magnified
on Saturday night.
</item 67>

<item 68>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“The doctors may have other plans, but he’s doing alright,” Abdelfattah said. “Obviously he was out for the rest of the game and he’s in pain a little bit, but
we’ll see.”

Goulding’s golden touch too much for Breakers

Veteran shooter Chris Goulding has fired a series of clutch three-pointers to pilot NBL ladder leaders Melbourne United to a 97-88 win over New Zealand.

Goulding had a quiet first half but drove daggers into the visitors at John Cain Arena on Sunday when he found range after the main break.

In his 300th game for Melbourne, Goulding delivered his trademark when he twice launched crucial bombs to help his team edge ahead in the dying stages of a
tight contest.

Goulding’s fade-away from the corner gave United a game-high nine-point lead with three minutes left as he finished with 4-of-5 shooting from distance.
</item 68>

<item 69>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Despite being without star big Aron Baynes (suspended) and import guard Shannon Scott (hamstring), the Bullets soared to a second straight victory on Sunday and
snapped a quirky streak in the process.

It was their first win after seven straight losses in Sunday fixtures, in a run that dates back to last season.

Far more importantly, the result levelled Brisbane’s current season record at 4-4 after five rounds.

But it didn’t come easily as fourth-quarter specialists Tasmania trimmed a seven-point deficit back to a single point when Majok Deng drove to the rack with 34
seconds remaining.

Advertisement

The two sides traded blows in the dying stages and the JackJumpers almost sent the game to overtime when they pinched Sam McDaniel’s inbounds pass on the final
play.

Jack McVeigh got a hand in to force the turnover and Tasmania found Jordon Crawford, but his decent look at an attempted game-leveller rimmed out.
</item 69>

<item 70>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The United lead grew to 16 before the JackJumpers showed their trademark fight with a 14-2 run – but they couldn’t quite complete the job.

Harvey’s Hawks buzzer beater sinks Taipans

Illawarra Hawks import Tyler Harvey has sealed an 84-83 upset defeat of the Cairns Taipans with a clutch three-pointer in the final seconds at WIN Entertainment
Centre.

Staring down the possibility of a 1-5 start to the NBL season, last year’s wooden spooners looked intent on staging an ambush on Thursday night as captain Sam
Froling fired early.

But the Taipans chased down a 13-point lead and edged in front in the final quarter, with Patrick Miller (24 points, seven assists) leading the charge.

Advertisement

Miller’s jump shot in the final 10 seconds gave the Taipans a two-point lead that looked as though it could have been match-winning.
</item 70>

<item 71>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Rangers 5, Diamondbacks 0

8:43 p.m. 2-RUN HOME RUN — Corey Seager has broken this game open! The 2020 World Series MVP has three hits in this series, but they're all home runs. He takes
Kyle Nelson deep and gives the Rangers a commanding 5-0 lead.

COREY SEAGER STRIKES AGAIN 💥



📺: FOX pic.twitter.com/oKytcm45H6 — FOX Sports: MLB (@MLBONFOX) November 1, 2023

Rangers 3, Diamondbacks 0

8:39 p.m. 2-RUN TRIPLE — There goes Semien! He hits on just inside the foul line, and it takes an eternity for Lourdes Gurriel Jr. to get to it. Two runs score.
It's 3-0, Rangers.

2-run triple for Marcus Semien!



The @Rangers came out swinging!
</item 71>

<item 72>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Without a hand being laid on him, ex-NBA forward Cheatham pulled up sore changing direction early in the fourth quarter at Auckland’s Spark Arena on Thursday
night.

Cheatham was unable to put weight on his right leg as Breakers staff helped him from the floor.

He played no further part in the match, but there was no immediate update on his condition after full-time.

Advertisement

“It’s too early. We’ll pray for good news,” said Breakers coach Mody Maor.

Any long-term injury would be troublesome for last season’s runners-up, who have begun the campaign with only one win from four games.

Cheatham’s injury proved a turning point. The Breakers had maintained a slim lead for most of the night as Cheatham (17 points, eight rebounds) combined with
guard Will McDowell-White to strong effect.
</item 72>

<item 73>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Christian Walker had a chance to turn two but couldn't hold onto the ball. The Rangers have the bases loaded with one out.

8:59 p.m. — Lowe keeps the line moving and laces a single up the middle. The Rangers have two runners on with one out. This one could get even more out of hand
for Arizona. Kyle Nelson's night is done.

8:58 p.m. — Jung hits a one-out single for his second hit of the night.

8:53 p.m. — Thomas flies out to left to end the inning.

8:51 p.m. — Gurriel hits a two-out single to give the Diamondbacks a pulse. That'll bring up Alek Thomas.

8:49 p.m. — Heaney notches two quick outs to start the bottom of the second.

8:44 p.m. — The inning ends on a nice throw by Emmanuel Rivera, but what an inning it was for Texas with five runs.
</item 73>

<item 74>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Arizona scored six runs over the final two innings, putting together competitive at-bats despite the dramatic deficit and forcing the Rangers to go to closer
Jose Leclerc for the final out.

Still, it wasn't nearly enough. The Diamondbacks' best hope is that their late-game surge carries over to Game 5, where they will face elimination for the third
time in this postseason.

The Rangers just need one more win to become World Series champions, and they will have the right man on the mound in Game 5 as Nathan Eovaldi makes his sixth
start of the postseason. Zac Gallen, who was less than sharp in October but earned Cy Young consideration during the regular season, will start for the
Diamondbacks with everything on the line for Arizona.

MORE: How Bruce Bochy went from retirement to World Series in one year

The Sporting News tracked live scoring updates and highlights from Game 4 of the 2023 World Series between the Rangers and Diamondbacks.
</item 74>

<item 75>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Without Miller running the point, the home side’s offence suddenly looked shaky in the third term as Sydney briefly took the lead on a Hogg triple.

The undermanned Taipans responded with a 15-0 burst either side of three-quarter time, highlighted by Swedish young gun Bobi Klintman’s thunderous dunk down the
middle over rival Next Star Alex Toohey.

Down by 14, Sydney closed the gap to four points before unsportsmanlike fouls against Denzel Valentine and Makuach Maluach ended their revival.

Lamenting “the defensive mistakes that we made”, Kings coach Mahmoud Abdelfattah added: “They made more plays than we did, they got to the 50/50 balls and they
stuck together.

“Kudos to them, they deserved it.”
</item 75>

<item 76>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Hanks was pivotal for the Demons and starred with 26 disposals, seven tackles and one goal, while her partner in the midfield Olivia Purcell had 28 touches,
five clearances and five tackles. For the Roos, Ash Riddell led the disposal count with 31, while skipper Emma Kearney was crucial in defence. Melbourne’s
Alyssa Bannan is tackled by Lulu Pullar of the Kangaroos. Credit: Getty Hanks said with the state of competition, they just had to take it week by week, but it
was good to get the win. “We knew we were up against the best tonight,” said Hanks. “It’s a nice feeling tonight.”

Lions roar in big battle Meanwhile, on Saturday afternoon, the Brisbane Lions put an end to Adelaide’s unbeaten run with a three-point victory at Brighton Homes
Arena in a top-four thriller, 5.12 (42) to 6.3 (39), in a spectacle worthy of being a final.
</item 76>

<item 77>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
28 Diamondbacks at Rangers 8:03 p.m. Fox ARI 9, TEX 1 (Tied 1-1) Oct. 30 Rangers at Diamondbacks 8:03 p.m. Fox -- Oct. 31 Rangers at Diamondbacks 8:03 p.m. Fox
-- Nov. 1 Rangers at Diamondbacks* 8:03 p.m. Fox -- Nov. 3 Diamondbacks at Rangers* 8:03 p.m. Fox -- Nov. 4 Diamondbacks at Rangers* 8:03 p.m. Fox --

*If necessary
</item 77>

<item 78>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Stepping up were acting skipper Bul Kuol (24, played all 40 minutes) and big Sam Waardenburg (22) for the Snakes, who shrugged off last Thursday’s
buzzer-beating heartbreak in Illawarra by taking the NBL’s most prized scalp.

Advertisement

It shot them from fifth to third on the ladder, while sending the Kings slumping from second to fourth.

“We kept moving down the bench to bring guys in,” Cairns coach Adam Forde said. “Everyone responded.

“I’m super happy for the group that they could fill some short-term reward with a win at home.”
</item 78>

<item 79>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
The Rangers found themselves one strike away from a championship twice in 2011. They couldn't finish the job. On Wednesday night, they'll have a chance to
avenge that painful finish.

Texas found out before Tuesday's Game 4 that slugger Adolis Garcia's season was over, but a talented lineup didn't miss a beat. The Rangers erupted for five
runs in the second inning and five more in the third, bludgeoning Arizona's bullpen to take a 10-0 lead in the blink of an eye.

Corey Seager came through with his third home run of the series, while Marcus Semien drove in five of the Rangers' first 10 runs — including a three-run blast
that put Texas up by double digits.

FAGAN: Three things Diamondbacks need to flip script on World Series

The Diamondbacks have earned a reputation for not backing down, and to their credit, they didn't.
</item 79>

<item 80>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
The Rangers can win it all as early as tomorrow night at Chase Field.

The @Rangers are one win away from their first #WorldSeries title in franchise history 🤠 pic.twitter.com/cHb2I3ZBOS — FOX Sports: MLB (@MLBONFOX) November 1,
2023

Rangers 11, Diamondbacks 7

11:22 p.m. 2-RUN SINGLE — The Snakes are still alive. Gabriel Moreno singles in two more runs to trim the Rangers' lead to 11-7.

The Snakes are not going away quietly 👀



📺: FOX pic.twitter.com/4XtYSF0fZj — FOX Sports: MLB (@MLBONFOX) November 1, 2023

11:18 p.m. — Bochy is bringing in Jose Leclerc for the final out despite the six-run lead...

11:17 p.m.
</item 80>

<item 81>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 81>

<item 82>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Chris Smith put up 16 points, Josh Bannan 15 and Tyrell Harrison 12 to go with five rebounds and three blocks in just 14 minutes.

First-year Bullets coach Justin Schueller couldn’t have been prouder of his team.

Advertisement

“It was a big game for us having dropped four, with two marquees out and we knew we had to come in here and play near a perfect game,” he said.

“My initial thought is just pleasure in how the guys stuck to a plan, and found our advantages consistently throughout the night.”

The Wildcats missed 18 of their first 19 three-point attempts before making a couple midway through the last quarter from Keanu Pinder, who finished with 22
points and five rebounds.

He was a lone shining light for a Wildcats team, who have lost four straight. Again, superstar Bryce Cotton struggled to 10 points and five assists on 3-of-19
shooting.
</item 82>

<item 83>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 83>

<item 84>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
— Smith strikes out Carroll, and the Rangers are an out away from a 3-1 series lead.

11:14 p.m. — Marte swings at a pitch well out of the zone and strikes out.

11:13 p.m. — Perdomo singles into the gap and sends Lawlar over to third. Arizona has a little something brewing here.

11:12 p.m. — Lawlar with a very professional plate appearance for a player who was just called up in September. He starts the inning with a walk.

11:09 p.m. — It'll be rookie Jordan Lawlar facing Will Smith to lead off the inning.

11:07 p.m. — Chris Stratton warmed up to start the inning, but Bruce Bochy is making a change after seeing Pavin Smith enter as a pinch-hitter.

11:04 p.m. — Jung quickly grounds into an inning-ending double play.
</item 84>

<item 85>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The guys are going in the right direction. “I’m not disappointed, I’m stoked.”

Bullets shoot down weakened Wildcats

Advertisement

The Brisbane Bullets have shrugged off the absence of two important players to outwork the Wildcats at their home RAC Arena in Perth and produce a performance
full of heart for their 84-79 NBL win.

Both teams were 2-4 coming into Friday’s match and on losing slides, but it was the Wildcats who had the pressure building, even with the Bullets being without
key pair, the suspended Aron Baynes and injured Shannon Scott.

Brisbane scored the first seven points before opening up a 20-point lead in the third quarter and going on to win by five after holding out against a furious
late charge from the home team.

The team effort of the Bullets was outstanding as they got another 17 points from captain Nathan Sobey.
</item 85>

<item 86>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Robinson was one of three Hawks to foul out, alongside big men Sam Froling and Lachlan Olbrich.

Advertisement

Despite Illawarra’s shooting issues, United gave up more first half points (51) than in any other of their games this season.

The Hawks dominated the offensive glass (18-10) and punished United with 21 points from turnovers.

But they finished the day shooting at 38 per cent from the field to United’s 53, which made things difficult, especially when United locked the paint down in
the third quarter.

“It took us a little bit of time to wake up with their physicality and the way they got after it,” said United coach Dean Vickerman.

NBL champion Lual-Acuil returned from a wrist injury for the first minutes of his United comeback, while NBA hopeful Hukporti (21 points, 10 rebounds) shone for
the visitors.
</item 86>

<item 87>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
The Rangers continue to be unstoppable road warriors.

Texas picked up its ninth consecutive road victory in these playoffs after defeating the Diamondbacks in Game 3 of the World Series, 3-1.

The Rangers were powered by some two-out magic in the third inning, as Marcus Semien finally found his bat with an RBI single before Corey Seager demolished a
two-run home run 421 feet to right field.

Jon Gray got credited with the win after Max Scherzer was pulled from the contest with a lower back injury through three scoreless innings. Gray was fantastic
in a surprise relief appearance, throwing three scoreless innings with three strikeouts and just one hit allowed.

The Diamondbacks' offense struggled to string together hits and couldn't cash in when they had runners in scoring position. Even though they out-hit the
Rangers, Arizona's bats hardly put any pressure on Texas' pitching.
</item 87>

<item 88>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But the Hawks belted down the court and the ball found itself in Harvey’s hands for their final roll of the dice.

His deep three swished through the net with one second remaining, sending the home crowd into raptures and improving the Hawks to 2-4.

“I turned around to my assistants and I said, ‘That’s why we have him (Harvey)’,” said Hawks coach Jacob Jackomas.

“It’s good that people can be reminded how special he is at the end of the clock.

“In that space, he always gives us a chance.”

The Taipans are now 2-2 to start the season, having also dropped their two games against NBA sides as part of that league’s pre-season.

Advertisement

“It’s a tall order to expect us to win if we keep giving up second-chance points,” said Taipans coach Adam Forde.
</item 88>

<item 89>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
9:44 p.m. — Make it 6-for-6 for Nelson, who has coasted through two scoreless innings. Let's see if Arizona can put any pressure at all on the Rangers.

9:43 p.m. — Nelson gets the first two outs without incident. He's retired all five batters he's faced.

9:40 p.m. — Fan on the field! At least there's some action.

9:37 p.m. — Heaney retires Alek Thomas, so the Diamondbacks come away from that opportunity with only one run. They're going to need more.

Rangers 10, Diamondbacks 1

9:35 p.m. SACRIFICE FLY — Gurriel hits one a long way to right field, but it's caught near the warning track. Still, it's a run. Moreno scores to put Arizona on
the board.

9:33 p.m.
</item 89>

<item 90>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9. Jacksonville Jaguars 6-3 (4)

The Jaguars got an NFC wake-up call against the 49ers, who were much more prepared after a Week 9 bye than they were. Jacksonville needs to forget that debacle
and get back on track, stat. The team gets a division double vs. Tennessee and Houston before facing Cincinnati.

10. Dallas Cowboys 6-3 (11)

The Cowboys took advantage of their "no contest" vs. the Giants at home with Dak Prescott continuing to ball as a passer and runner. They need to keep the
defense and running game success, too, in bigger games going forward.

MORE: Inside the Cowboys historic offensive explosion vs. Giants

11.
</item 90>

<item 91>
World Series Game 2: Arizona Diamondbacks v Texas Rangers – live (The Guardian, 2023-10-28T23:00:28+00:00)
29 Oct 2023 02.33 GMT RBI single! (Carroll) Diamondbacks 7-1 Rangers Diamondbacks 7-1 Rangers, top 8th Carroll singles here and scores Perdomo! The Diamondbacks
are just teeing off on these relievers.



29 Oct 2023 02.31 GMT Two-run single! (Marte) Diamondbacks 6-1 Rangers Diamondbacks 6-1 Rangers, top 8th This is more like the Rangers bullpen I was expecting.
Ketel Marte is up next with the bases loaded and a chance to make the rest of this game entirely irrelevant. Pérez’s first pitch is yet another ball, this one
that barely misses the ground. 1-0. Marte takes a huge swing and the next pitch, he doesn’t make contact. However, on the next pitch, he knocks in a base hit
that’s a two-run, two-out single and the Diamondbacks have a five-run lead!
</item 91>

<item 92>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Sobey and Smith then knocked down shots as the Bullets lead ballooned to 20.

The Wildcats did make a charge with a Pinder-inspired 11-0 run in the fourth to cut the deficit to three, but the Bullets answered with strong inside finishes
from Harrison and Bannan to secure the win.

United flex muscles in road win over JackJumpers

Melbourne United have had to withstand a hot start and finish from the Tasmania JackJumpers but in between were outstanding on the way to a 90-82 victory in the
NBL.

Advertisement

The league’s top two teams locked horns at a sold out MyState Bank Arena on Friday night and it was the JackJumpers who raced out of the blocks with a 17-4
start, but United took control from there.
</item 92>

<item 93>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Advertisement

“We can’t not talk about CG’s couple of threes down the stretch,” coach Dean Vickerman told reporters post-game. “They were pretty incredible.”

Jo Lual-Acuil (16 points) and import Ian Clark (18) also came up with big plays in the final period as Melbourne outscored their opponents 33-25 with the game
on the line.

Lual-Acuil shone in his second game back from a long-term wrist injury, and Clark’s wide-open three restored United’s nine-point buffer with 38 seconds left to
play. Such was Goulding’s confidence in his teammate, he turned around to celebrate the game-sealing bucket with the home crowd before Clark had even released
his shot.

Melbourne had five scorers in double figures in an even team performance in the absence of injured guard Matthew Dellavedova, who has missed two games because
of his latest concussion setback.
</item 93>

<item 94>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Ex-Taipan Kouat Noi (18 points) top-scored for the visitors, while Jaylen Adams (13) and DJ Hogg (13) – playing against his former side in his long-awaited
debut for his new team following a shoulder injury – got on the scoresheet, but both shot under 30 per cent, embodying the Kings’ rocky evening.

The two-time reigning champions were all at sea early, out-rebounded 18-6 in the opening term while missing their first 15 three-pointers.

Cairns boosted their 26-20 quarter-time buffer to 12 points shortly before Miller’s night ended in stomach-churning style.

Advertisement

The injury left their backcourt stocks further depleted, with Taran Armstrong (foot) and captain Tahjere McCall (family reasons) already missing.
</item 94>

<item 95>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
In Adelaide, the 36ers responded to a disappointing loss on Thursday night, also at home to the Phoenix, and handed the Wildcats a third straight defeat with
their 11-point win.

Sitting in bottom spot with one win from five games coming into the macth, the players got together to draw a line in the sand.

It certainly worked and their effort, energy and desperation was dramatically improved against Perth, but so was the execution and a focus on getting Isaac
Humphries more involved inside.

He put up 26 points and eight rebounds in a match-winning performance but the shortening of the rotation to use just eight players from coach CJ Bruton worked
too.

Advertisement

“We met as a playing group yesterday and talked through a lot of stuff,” Humphries said.

“We just wanted to come out today and put into action everything we talked about.

“We executed what the coaches needed from us tonight and we executed what we, as a playing group, wanted to execute as well.”
</item 95>

<item 96>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
7:35 p.m. — For all of the Rangers' losses, a win tonight with Nathan Eovaldi and Jordan Montgomery in line to start the next two games would put Texas in a
terrific position.

7:18 p.m. — Max Scherzer is also off the World Series roster for Texas. He's been replaced by Brock Burke.

7:07 p.m. — It's being reported that Adolis Garcia is off the World Series roster. His season is over. Ezequiel Duran has been added to the roster.

6:45 p.m. — Look who spent some time together before Game 4? The Diamondbacks are earning Chris Russo's trust.

6:25 p.m. — Bruce Bochy said he's anticipating making a roster decision on Adolis Garcia tonight. Garcia has a moderate oblique strain.

5:10 p.m. — Adolis Garcia is not in the lineup for Texas.
</item 96>

<item 97>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Crows elected not to challenge the decision. ‘Horror week’: Bombers stunned by struggling, controversy-hit Eagles West Coast withstood a late-term surge
from Essendon to narrowly claim their second win of the AFLW season.

Fresh off a controversy caused by comments their coach Michael Prior made following last week’s 70-point loss to Melbourne, the Eagles prevailed 3.5 (23) to 2.7
(19) in a slugfest at the appropriately named Windy Hill on Sunday. West Coast’s Kate Bartlett (left) and Isabella Lewis celebrate Sunday’s surprise win over
Essendon flanked by their teammates. Credit: Getty Prior had blasted the decision to pit strugglers West Coast against defending premiers Melbourne, saying:
“How we play that team is beyond me. That is what you get when you get fixturing like that.” He later backtracked on the criticism, but his comments did put
added pressure and focus on his team this week. Michael Prior guided West Coast to a gutsy win.
</item 97>

<item 98>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
— Jung gets moved over to third on a chopper by Heim, who is out at first. Leody Taveras has a two-out RBI opportunity.

8:31 p.m. — Miguel Castro is in to face Heim with one out and Jung at second.

8:29 p.m. — Mantiply quickly strikes out Lowe for the first out of the inning, and that will be the end of his day.

8:27 p.m. — Keeping Mantiply in to face Josh Jung was not successful. Jung belts a double into the gap and gives Texas a leadoff runner in scoring position.
Mantiply will stay in to face Nathaniel Lowe.

8:25 p.m. — The Diamondbacks are sticking with Mantiply to start the second inning.

8:23 p.m. — Heim nabs Marte on the stolen base attempt! That'll end the inning.
</item 98>

<item 99>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
At the other end, Browne racked up nine points and four assists in the opening period, which ended with scores tied at 28-28.

Vasiljevic’s third triple gave Adelaide a slender 40-39 lead, before the visitors rattled off the next 15 points and led 54-42 at half-time.

After Browne led Phoenix’s early charge, it was fellow imports Williams and Cummings who, in the second stanza, troubled the Sixers, who looked all at sea
offensively and passive defensively.

After Phoenix’s lead ballooned to 17 points midway through the third, experienced 36ers reserves Sunday Dech and Jason Cadee helped the home side briefly trim
the deficit to single digits.
</item 99>

<item 100>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
The @Rangers came out swinging!



📺: FOX pic.twitter.com/Je43lpXXMi — FOX Sports: MLB (@MLBONFOX) November 1, 2023

8:37 p.m. — Taveras walks, and Travis Jankowski quickly singles up the middle to give the Rangers runners on first and second with two outs. Semien is up with a
chance to extend Texas' lead.

Rangers 1, Diamondbacks 0

8:35 p.m. WILD PITCH — Jung scores on a ball that gets away from Moreno! Castro's pitch in the dirt took a bad bounce, and there was no play at the plate. The
Rangers lead early.

The @Rangers strike first in Game 4!



📺: FOX pic.twitter.com/E5FCruQby7 — FOX Sports: MLB (@MLBONFOX) November 1, 2023

8:33 p.m.
</item 100>
=== END ITEM id=a94288701fc4b6b4 ===
