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

This file has 2 items.

=== ITEM id=b1c9d45fc337a318 ===
Gold sentences and facts:
Sentence 0: “It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game in junior — it indicated he was going to be able to do that at some point in the NHL.
  Fact 0.0: Davidson said the player was not playing in the NHL in the last few years.
  Fact 0.1: Davidson said the level of performance showed the player could control a game and dominate a game in junior.
  Fact 0.2: Davidson said this performance indicated the player was going to be able to do that at some point in the NHL.
Sentence 1: There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala, among many others.
  Fact 1.0: The player helped guide more than a few future NHL stalwarts at that place.
  Fact 1.1: Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala were among the future NHL stalwarts guided there.
  Fact 1.2: Many other future NHL stalwarts besides those named were also guided there.

Retrieved items:
<item 1>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.
</item 1>

<item 2>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“It’s a lot of different angles he can give you.”

Sabres first-round pick Zach Benson, who has played on teams with and against Bedard and skates with him in the summers, says it’s Bedard’s ability to set up
the shot with his agility pre-shot that stands out to him.

“He can move left to right like no one I’ve seen before,” Benson said.

“It’s just trying to get out, face him, and hope it hits ya,” said Scott Ratzlaff, Seattle Thunderbirds goalie and Buffalo Sabres prospect, with a smile. “He
can shoot from anywhere and he’s lethal from anywhere. You’ve just always got to be ready just in case he shoots it. And then he’s got a really good toe drag
release, so it’s watching for that and making sure you’re lined up.”

A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting.
</item 2>

<item 3>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
ET, 7 p.m. CT in Chicago.

The game is being played at PPG Paints Arena in Pittsburgh, Pa.

MORE: Ranking the NHL's top 50 players for the 2023-24 season

What channel is Blackhawks vs. Penguins on today?

TV Channel (U.S): ESPN

ESPN TV Channel (Canada): Sportsnet, TVA Sports

Sportsnet, TVA Sports Live stream (U.S.): ESPN+, Fubo

ESPN+, Fubo Live stream (Canada): Sportsnet+

ESPN has coverage of the Blackhawks vs. Penguins game on Tuesday evening. The Pittsburgh-Chicago contest is one of three games that the network has on opening
night.

ESPN's "A-team" will be on the call for Bedard's debut, as Sean McDonough (play-by-play), Ray Ferraro (analyst) and Emily Kaplan (reporter) will be the
broadcast team for Blackhawks vs. Penguins.
</item 3>

<item 4>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.
</item 4>

<item 5>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.
</item 5>

<item 6>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Rust gets the Penguins their first of the season 🚨 pic.twitter.com/RKNyaJT0LD — B/R Open Ice (@BR_OpenIce) October 11, 2023

8:28 p.m. — It's a strong showing from the top PP unit for the Blackhawks, but Bedard and Co. can't capitalize on the man advantage. Chicago had four shots on
goal, including a Bedard one-timer that was stopped by the pad of Mrazek.

8:26 p.m. — Penguins forward Lars Eller is called for goalie interference, as he clips the leg of Petr Mrazek in the crease. The Blackhawks head to the first
power play of the game about five minutes into the period.

8:23 p.m. — Through the first four minutes, it's been a back-and-forth affair. Neither team have been able to sustain much pressure. Most of Chicago's shot
attempts have been from the outside, making it easy on Tristan Jarry.
</item 6>

<item 7>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“So many players pull the puck in but can’t get the puck to release from under their body. His footwork is so underrated in that aspect. If you watch his front
leg, at times it’s literally in the air at release — most have their back leg in the air at release. When he transfers his weight, he actually clears space for
his hands to get tighter because he’s not afraid to actually lunge into the shot.”

A fourth shooting coach, who hasn’t worked with Bedard and also requested anonymity because he works for another NHL club, expressed a little more hesitancy.

Advertisement

“I’m in the minority I’m sure, but I am very curious to see how well his trademark shooting style works at this level,” the coach said. “He really likes to pull
pucks right into his feet.
</item 7>

<item 8>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Bedard was as advertised in his NHL debut. The Chicago rookie led all players on his club in shots on goal, recording five in his first game on 11 shot
attempts. His 21:29 time on ice was second to only Seth Jones, as Chicago head coach Luke Richardson did not hold the 18-year-old back. He did struggle in the
faceoff dot, winning just two of 13 draws, but it was a strong first impression at the NHL level.

For the Penguins, the third period struggles from last season have seemingly carried over to start this year. Last year, Pittsburgh finished tied for the most
losses after taking a lead into the third period. The crew followed that trend on Tuesday night, letting a 2-1 lead to start the final frame slip away.

Bedard and the Blackhawks are back in action on national TV tomorrow, as Chicago heads to Boston for a Wednesday night date with the Bruins.

MORE: Watch Blackhawks vs.
</item 8>

<item 9>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
While almost all players point their toes at the net, set the puck at a right angle, load their shot and “drag it in and then snap it off,” Turk says, on a line
of 40 to 50 degrees behind their body or at least level with their heels, Bedard brought the puck in “completely lateral.”

Instead of drawing the puck in at a 45-degree angle, he drew it at 180 degrees “right across to the plane of his spine.” Sometimes it even came in at a negative
angle. It also happened in a smaller space than it does for other shooters. That lateral inward pull was “condensed — it’s compacted.”

Turk says that Vladimir Tarasenko, whom Turk calls one of the NHL’s quickest shooters, takes 18 inches to pull the puck in. Bedard did it all in 12 square
inches — a one-foot box.

Turk’s reaction?
</item 9>

<item 10>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Korchinski was the No. 7 overall pick in the 2022 draft, one of three first-round selections by Chicago that year. He played his junior career with the Seattle
Thunderbirds of the WHL.

7:36 p.m. — You want to talk confidence? How about going without a bucket in warmups? Bedard is rocking the no-helmet look ahead of his NHL debut.

Connor Bedard going with the no-helmet look during pregame warmups. #Blackhawks pic.twitter.com/qrr93vlMsh — Charlie Roumeliotis (@CRoumeliotis) October 10,
2023

7:33 p.m. — A stick may be important when playing the game of hockey.

Note to self:

Don't forget your stick ahead of your NHL debut. pic.twitter.com/guzVprRXfY — Sportsnet (@Sportsnet) October 10, 2023

7:02 p.m.
</item 10>

<item 11>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Erik Karlsson with the moves 👀#LetsGoPens | #NHLFaceoff pic.twitter.com/GCWVIGnAS2 — Hockey Daily 365 l NHL Highlights & News (@HockeyDaily365) October 11, 2023

8:41 p.m. — Pittsburgh heads to its first power play of the game, as Wyatt Kaiser trips up Malkin in the neutral zone. This will be the first time seeing Erik
Karlsson running the power play for the Penguins, taking Letang's place on the top unit with Crosby, Malkin, Guentzel and Rickard Rakell.

8:36 p.m. — Evgeni Malkin makes a beautiful saucer pass to Jake Guentzel cutting the net, but the Penguins winger could not elevate the puck over the shoulder
of Mrazek. At the other end, Taylor Raddysh took a feed from behind the net, but put it right in the midsection of Jarry.

8:34 p.m.
</item 11>

<item 12>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
📺: ESPN, ESPN+ pic.twitter.com/Qsk9Jgfp1c — Pittsburgh Penguins (@penguins) October 10, 2023

#Blackhawks Opening Night lineup vs. Penguins:

Hall-Bedard-Donato

T. Johnson-Reichel-Raddysh

Katchouk-Guttman-Athanasiou

Perry-Dickinson-Foligno



Vlasic-Jones

Korchinski-Murphy

Tinordi-Kaiser



Mrazek

Söderblom



Extras: Entwistle, R. Johnson, Zaitsev — Charlie Roumeliotis (@CRoumeliotis) October 10, 2023

7:50 p.m. — Bedard isn't the only rookie making their debut tonight. Blackhawks defenseman Kevin Korchinski is playing in his first NHL game tonight. Korchinski
was the No.
</item 12>

<item 13>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
You just never how quickly.

“It’s a big adjustment, especially when you’re playing against men and being focused on by the other team every single night. I’m excited for him, but I don’t
necessarily feel like it’s a huge surprise how quickly he’s started to really find his groove at the NHL level. But it’s exciting to watch.”

(Top photo: Ethan Miller / Getty Images)
</item 13>

<item 14>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
But just because injuries might take some of the juice out of these games doesn't mean that SuperDraft isn't giving us plenty of reasons to care. There are some
snazzy looking props out there for us to make our bottom line look at lot better than it did earlier this morning.

SuperDraft users can win 3X the amount wagered if their two-legged player prop parlay hits. If all four of the props below hit, users will win 10x their entry.
In other words, a $20 entry fee will yield a $200 payout. the more legs you add to your player prop parlay, the larger your potential payout becomes, but as the
long odds indicate, completing a clean sweep is challenging.

Below, we'll break down four of our favorite player props for Tuesday night's NHL slate. Users don't necessarily have to parlay all four picks together, though.
</item 14>

<item 15>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Daniel Sprong, RW, Red Wings @ Rangers — OVER 2.5 shots on goal

Despite not being featured in the top six, Sprong is holding down the fort on the third line, averaging five shot attempts per game through the first 12. He's
hit the over in five straight, with 18 SOG compared to just eight combined for his two linemates. Sprong does see some work on the second power play unit, but
is clearly willing to fire a lot of pucks at the net with his shoot-first mentality on the third line. With the Rangers still figuring out how to deal with the
loss of Fox, there should be some opportunities there for Sprong to find at least three shots tonight.

3. Jesper Bratt, LW, Devils @ Avalanche — UNDER 0.5 assists

Bratt comes into Tuesday night's action with seven goals and 11 assists in 11 games.
</item 15>

<item 16>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“Well who the f— are these guys?” Turk finally asked him.

“Turky, it’s Connor Bedard,” Landry said.

“Who’s Connor Bedard?” Turk asked.

After the second session that same week, Turk didn’t work with Bedard again until a stick manufacturer tried to pitch Bedard, who asked if Turk could be there
when he test-drove the sticks.

By then, Turk knew the Bedard name and came away from another session with another takeaway, picking up on Bedard’s eye contact and ability to change his mind
mid-shot.

“What makes him unique is that he can be selective with it,” Turk said.

When he looks back on that first blind introduction, Turk laughs.

“You know, most, when you’re in a non-pressure, non-stressful situation, you’ll play around with the puck a lot and then you’ll take your shot because I’m
telling them to take their time,” he said.
</item 16>

<item 17>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury. Hughes
went down after slamming into the boards against the Blues, forcing New Jersey to change things on the fly. The changes have directly impacted Bratt, who was
featured on the top line alongside Hughes. With fellow center Hischier also out with an injury, leaving the Devils fairly weak down the middle and trusting
Michael McLeod to hold down the fort in between Bratt and Tyler Toffoli. As a result, Bratt has been held without an assist in each of the last two games for a
team that's missing a key contributor on offense. And it'll need him to look to score rather than play the role of playmaker. Until the Devils figure things
out, we can't trust this team to do anything positive in the offensive zone.

Win BIG with SuperDraft!
</item 17>

<item 18>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The Wild will certainly get a boost in the standings once Fleury and Gustavsson improve their play between the pipes, a likely scenario given their stellar
performance last season. However, they again have the misfortune of being in quite a competitive division. The Avalanche and Stars occupy first and second place
as expected powerhouses, while the Blues, Predators and Coyotes all sit at or above the .500 mark and, especially in Nashville’s case, have posted decent
underlying numbers.

Hynes’ main job is simply to restore confidence in his group. This roster has the ability to play better — especially Kaprizov, who, despite the okay point
production, has looked like a complete non-factor on some nights. The team also hopes a significant change behind the bench can boost their league-worst penalty
kill, which is operating at just 66.7%. Penalty killing was an area of strength for Nashville last season, who finished tied for fourth in the NHL with an 82.6%
success rate.
</item 18>

<item 19>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
What do the Giants’ and Vikings’ ownerships think?

“The best first year can be that 7-10 season where you show some promise,” a veteran coach with experience on a half-dozen teams said. “The Giants won a bunch
of games early, people were heaping praise on the head coach, people get the expectation of what the quarterback can be. But oftentimes, you know different
internally because you know the house you are living in. You are inside the walls every single day and you know where the roof leaks, where the foundation is
cracked.”

Buffalo managed such a situation well. The Bills surprisingly reached the playoffs in their first season under Sean McDermott, a euphoric moment after a 17-year
playoff drought. They went 6-10 the next year, then began a steady rise behind Josh Allen and a top defense. There isn’t a Josh Allen on the Giants or Vikings.
</item 19>

<item 20>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Consistent high-level performances

AFL player comparison: Andrew Brayshaw (Fremantle)

Ari Schoenmaker

Pronunciation: Show-en-maker

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 07/01/05

Height: 194cm

Left-footed tall defender who reads the game exceptionally well, is a good interceptor and prodigious kick of the ball. Had a very impressive back half of the
year, which included being selected for the Allies in their final two matches of the National Championships. He also showed the ability to play on the wing and
averaged 27 disposals at 75 percent efficiency as well as 5.9 marks and 9.6 rebound 50s in his eight matches with the Tasmania Devils in the Coates Talent
League to be one of their most impressive performers.
</item 20>

<item 21>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
First game, first point, first win.

Connor Bedard's debut did not disappoint on Tuesday, as the Chicago rookie helped the Blackhawks to a 4-2 win over the Penguins on opening night.

The Blackhawks rallied from a 2-0 deficit, scoring four unanswered goals to pick up the victory in Pittsburgh.

Bryan Rust opened the scoring for the Penguins in the first period, and a Sidney Crosby goal in the second doubled the Pittsburgh lead. However, Bedard helped
set up a Ryan Donato goal in the middle frame to get the Blackhawks on the board, picking up his first NHL point in the process.

In the third period, Cole Guttman potted the equalizer for Chicago early in the frame. With less than five minutes left in regulation, Jason Dickinson fired
home a shot from the slot that gave Chicago its first lead of the night. A Nick Foligno empty net goal sealed the deal for the Blackhawks.

Bedard was as advertised in his NHL debut.
</item 21>

<item 22>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— GOAL! Ryan Donato slips home a rebound from the top of the crease to put Chicago on the board, and Bedard has his first NHL point, picking up a secondary
assist on the tally. Alex Vlasic walks in off the wall and puts a low shot on Jarry that is saved, but Donato pulls the rebound around and by Jarry's leg with
4:23 left in the second.

🚨first NHL assist for Connor Bedard. the full sequence starting with his zone entry pic.twitter.com/JqJMhchiTN — Shayna (@hayyyshayyy) October 11, 2023

Ryan Donato puts home the juicy rebound to give Connor Bedard his first NHL point!#Blackhawks | #NHLFaceoff pic.twitter.com/JbjVPvHVLE — Hockey Daily 365 l NHL
Highlights & News (@HockeyDaily365) October 11, 2023

9:46 p.m.
</item 22>

<item 23>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
It's 2-2 with 9:55 left in regulation.

Tie game with less than half the period to go 👀 pic.twitter.com/uqB1Z2Bnol — B/R Open Ice (@BR_OpenIce) October 11, 2023

10:24 p.m. — The Blackhawks can't take advantage of a power play again, dropping to 0 for 4 tonight on the PP. Pittsburgh remains ahead 2-1 with 13:43 left in
the period.

10:20 p.m. — Pittsburgh kills off the Nieto penalty, but seconds after going back to even strength, Letang picks up a cross-checking minor. Chicago goes right
back to the man advantage.

10:19 p.m. — Corey Perry had not just one, but two point-blank chances in front of the Pittsburgh net, but he put the puck right into the midsection of Jarry
both times.

10:17 p.m.
</item 23>

<item 24>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting. Patrick Kane would stay out with the young guys, including Alex
DeBrincat, well after practice to play shooting games. But even they had a limit.

Bedard doesn’t have much of a limit. It’s common for him to be on the ice an hour after practice taking shots from all over the ice. He even got a few other
rookies, especially defenseman Kevin Korchinski, to join him on a regular basis. As the media waits for Bedard at his dressing room stall for interviews, he
shoots and shoots and shoots.

Advertisement

Blackhawks coach Luke Richardson has been around long enough to see that in a few other special players.

“I know Jaromir Jagr did this for a long time,” Richardson said. “Even at the end of his career — and he skated around and he had the weights around the bottom
of his ankles, practiced full practices like that. It’s happened.
</item 24>

<item 25>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Users don't necessarily have to parlay all four picks together, though. Instead, you could make separate two-legged parlays to increase your win probability or
attempt to hit it big if all four picks come through.

Win BIG with SuperDraft! Get a free $10 deposit when you use promo code "TSN"!

Best NHL prop bets today: SuperDraft player prop picks for Wednesday

*10x potential payout if users parlay all four correct picks

1. Artemi Panarin, LW, Rangers vs. Red Wings — OVER 3 shots on goal

Panarin has registered at least three shots on goal in nine of 11 games this season, going over three in six of those contests. The winger has been up to his
usual antics in the new season, but is also seeing some extra ice time under new head coach Peter Laviolette.
</item 25>

<item 26>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.
</item 26>

<item 27>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Tuesday night's NHL national games feature an Original Six matchup at Madison Square Garden in addition to a battle between the team that left and the team that
took its place. The Rangers and Red Wings will face off at the world's most famous arena as both squads have jumped out to hot starts in the new season. Dylan
Larkin is a game-time decision for the Motor City while injuries to the Blueshirts' Adam Fox, Filip Chytil and Igor Shesterkin are making sure some of this
matchup's biggest stars will not be watching in uniform. Regardless, there is still plenty of star power on Broadway in what should feature plenty of offense.

In the late game, the Devils return to Colorado -- the place they left in order to move to New Jersey -- for a showdown with the Avalanche. Again, injuries are
a big part of the storyline here, thanks to Jack Hughes and Nico Hischier out until further notice.
</item 27>

<item 28>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
However, the Blackhawks are challenging the play for offside. Marcus Pettersson had to reach for the puck along the blue line and it may have come out of the
zone.

9:27 p.m. — Jarry keeps the shutout going with two real strong saves on Johnson. A shot from the point was tipped by an unmanned Johnson at the top of the
crease. When that didn't go, he got a second whack at the loose puck, but Jarry stuck out the leg to make the stop.

9:23 p.m. — Reilly Smith, who was traded to the Penguins this offseason from the Golden Knights, is denied by the glove of Mrazek off the rush. Smith had a puck
trickle to him on the right side, but he shot it right into the mitt of the Blackhawks goaltender.
</item 28>

<item 29>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
These game plans were conservative enough to signal a potential in-season quarterback change. Denver faces Green Bay and Kansas City over the next two weeks
before heading into its Week 9 bye.

5. You play to win the game, but can you win too quickly? The Giants and Vikings are test cases.

The New York Giants and Minnesota Vikings exceeded expectations under first-year head coaches in 2022. Both reached the playoffs. Both are struggling in their
second seasons, facing difficult questions about the future. Would they have been better off enjoying only moderate success early? That could depend on how well
the football leadership has communicated with ownership.

Every head coach and general manager wants to win as much as possible, but the goal is to build for the long term, which can require time to do things with
sustainability in mind. Bad teams that win right away can reset expectations for fans and ownership, making it less practical to take one step backward in the
short term for a better shot at taking two steps forward down the line.
</item 29>

<item 30>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
pic.twitter.com/6TWSCfXUlR — Steven Ellis (@SEllisHockey) October 11, 2023

8:47 p.m. — Noel Acciari and Tyler Johnson collide in the neutral zone, and Johnson is left leaking from the bridge of his nose. Initially Acciari is assessed a
double minor for high-sticking, but after reviewing the play, no penalty was killed. The officials deemed that the stick was not above the shoulders, therefore,
no infraction.

8:46 p.m. — The Blackhawks kill off the Kaiser minor. Pittsburgh managed to get three shots on Mrazek, but could not build on its lead.

8:43 p.m. — Karlsson dances by Cole Guttman at the blue line, walks into the slot and lets one rip, but Mrazek stares it down the whole way, making the glove
stop on the Penguins defenseman.
</item 30>

<item 31>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The Rangers head into this matchup with a makeshift lineup thanks to those injuries that is forcing them to adjust. The absences of Fox and Chytil are a big
deal, however, as Panarin loses Fox, who quarterbacks the power play, and Chytil, who centered his line at even strength. Panarin is still all over the ice for
the Blueshirts though and is even seeing 18.8 percent of his shifts start in the offensive zone, a welcome sight after the last two seasons with Gerard Gallant
behind the bench. The Red Wings have allowed the 11th most SOG this season (371) and the fourth most power play opportunities (46). If the Rangers can play with
the man advantage, Panarin should hit this number with ease, but he's enough of a factor at even strength that we like the OVER here anyway.

MORE SUPERDRAFT DFS & FANTASY PROPS: How to play SuperDraft

2.
</item 31>

<item 32>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Obviously, the goalies are good, you got to hit your spot.”

Blackhawks goalie Arvid Söderblom would certainly call Bedard a shooter. Söderblom and Petr Mrazek have faced more shots from Bedard than any NHL goalies this
season. The challenge for them is that Bedard doesn’t just shoot one way. He’s unpredictable.

“When you have that type of shot, quick release, moves his body and shifts angles on the puck, it makes it harder on the goalies,” Söderblom said. “You just saw
the two goals (on Sunday against the Florida Panthers), both were quick releases. He finds that open net. He’s a pure goal scorer. It’s fun to have on the team
and face him every day and see him take steps, too.”

While goalies are trying to figure out Bedard, he’s been putting in the time to do the same with them. For one, he learned to utilize his teammates more.
</item 32>

<item 33>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
If there’s one thing Tim Turk knows, it’s shooting. A self-described “shooting and scoring coach,” Turk has more than two decades of experience working with NHL
stars, national federations and half a dozen NHL teams.

Turk had no idea who Bedard was when he fielded a call from agent Greg Landry of Newport Sports asking him what he was doing tomorrow.

“Tomorrow?” he said. “I’m booked up a year in advance.”

“We have these two players coming in from out west and we want you to see them,” Landry said.

“Listen, I’m booked up,” Turk said again.

“You have to see them,” Landry insisted.

After moving around his schedule, Turk found himself at Gary Roberts’ personal rink in Uxbridge, Ontario.

“Hey guys, I’m Tim Turk,” he said as he stepped into the small locker room.

“My name’s Nate,” said the redhead, extending a hand.

“My name’s Connor,” said the other, standing shorter and extending his.
</item 33>

<item 34>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
MORE: Watch Blackhawks vs. Penguins live on Fubo (free trial)

The Sporting News provided updates and highlights from Connor Bedard's NHL debut in the Blackhawks vs. Penguins game.

Blackhawks vs. Penguins score

1 2 3 F Blackhawks 0 1 3 4 Penguins 1 1 0 2

Blackhawks vs. Penguins live updates, highlights from Connor Bedard's NHL debut

All times Eastern

Final: Blackhawks 4, Penguins 2

10:50 p.m. — The Blackhawks score four unanswered goals to come from behind and take down the Penguins on opening night. Bedard was as advertised in his debut,
picking up his first NHL point in the process. The third period struggles continue over from last season for the Penguins.

10:48 p.m. — Jarry back to the Pittsburgh bench again for the extra attacker.

Blackhawks 4, Penguins 2

10:48 p.m. — GOAL!
</item 34>

<item 35>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
Injuries and goaltending have certainly played a factor in their poor play — captain Jared Spurgeon missed the first 13 games of the season, and Filip
Gustavsson’s and Marc-André Fleury’s combined .878 SV% is near the bottom of the league.

They’ve likely been unlucky, too. The Wild have controlled a slim majority of scoring chances and a strong majority of high-danger chances during 5-on-5 play, a
common theme for an organization that’s usually one of the more defensively stout in the league. But the culture around the team is evidently turning sour
quickly, Kirill Kaprizov is on pace for a career-low 26 goals and has a team-worst -10 rating, and most of their depth defensemen have been significant
liabilities. This roster is not all that different from the one that’s finished top-three in the Central Division for three straight seasons, and it’s certainly
one that still has playoff aspirations.

Minnesota has not won since Nov. 7 against the Islanders and is winless in seven straight.
</item 35>

<item 36>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Win BIG with SuperDraft! Get a free $10 deposit when you use promo code "TSN"!

4. Cale Makar, D, Avalanche vs. Devils — OVER 2.5 shots on goal

Over this number in each of the last two, fans will want to see Makar shoot the puck a little more after going under in five of the previous six before his
recent hot streak. Facing a Devils team that is reeling thanks to injuries should help Makar's cause with Colorado able to control the play in the Mile High
City. He's the team's top defenseman and a key piece to their power play. Barring something unforeseen, Makar will have plenty of opportunities to find three
shots before the 60 minutes on the clock run out. And with 5.9 shot attempts per game, it's only a matter of time before they start finding their way on goal.
</item 36>

<item 37>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The 59-year-old Evason began his coaching career with the WHL’s Calgary Hitmen as an assistant in 1998-99 after finishing out his playing career in Germany,
working his way up the junior hockey ranks over the next handful of seasons. His first NHL job came courtesy of an assistant role with the Capitals immediately
after the 2004-05 lockout, coaching Alex Ovechkin in his first seven NHL seasons.

After departing Washington in 2012, Evason spent the next several seasons working in the Predators organization as the head coach of their AHL affiliate, the
Milwaukee Admirals. There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor
Arvidsson, and Kevin Fiala, among many others. After joining the Wild, he earned a promotion to head coach within two seasons when the Wild fired Bruce Boudreau
late in the 2019-20 campaign.
</item 37>

<item 38>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“He really likes to pull pucks right into his feet. I really like his mechanics with his hands off his body and whippy stick — but I feel this habit makes it
easier for elite level ‘D’ to get stick on puck or in the lanes to block shots. Finding space and time to let that shot go and get clean looks, I think, will be
an adjustment.”

For goalies, it’s the angle change that gets them.

Moose Jaw Warriors goalie Jackson Unger, whom Bedard scored on last year, said the toe-drag release, in particular, is a challenge.

“He changes the angle so quickly that, as a goalie, you have to adjust to it, but when he does it so fast, it’s easier said than done,” Unger said. “It’s a lot
of different angles he can give you.”
</item 38>

<item 39>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Most of Chicago's shot attempts have been from the outside, making it easy on Tristan Jarry.

8:19 p.m. — It was an uneventful first shift for Bedard, as the Penguins controlled most of the play in the Blackhawks' zone.

8:17 p.m. — Bedard and Crosby square up for the opening face off, and Bedard's debut is underway!

Sidney Crosby 🆚 Connor Bedard on the draw to start this one off.



What a moment. pic.twitter.com/4BPboh8jdp — Sportsnet (@Sportsnet) October 11, 2023

Pregame

7:58 p.m. — Here is how the two sides are lining up for the season opener.

Almost time for hockey!
</item 39>

<item 40>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“My name’s Connor,” said the other, standing shorter and extending his.

“All right, well we’re just going to go out and I’m going to do an evaluation on you and do some shooting and have some fun, and I’ll make suggestions but I’m
not here to change anything,” Turk said, giving the two boys his usual spiel and thinking nothing of it.

That evaluation is what he calls an “NHL protocol observation assessment,” which runs new clients through specific drills that allow him to visualize their
shooting actions from a technical standpoint (he calls each player’s shot “like a fingerprint”).

That starts with Turk throwing some pucks down in the slot and simply asking them to shoot them stationary so he can watch their body formation, hip and
shoulder positions, spine angles and puck preparation and positioning, paying particular attention to where the puck starts in their stance versus where it
finishes.

Advertisement

On that day, Nate (Danielson, the Red Wings’ 2023 No.
</item 40>

<item 41>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
A strong showing so far from Bedard, who has not shied away from putting the puck on net. He had four shot attempts in the opening frame, with two on net. The
Blackhawks out-shot the Penguins 17-13, but Pittsburgh holds the lead.

8:51 p.m. — An excellent shift by Bedard nearly results in his first goal. Initially, Bedard shows off his elite release by zipping a shot from long-range, but
it's turned aside by Jarry. Later in the shift, the rookie gets two whacks at a loose puck in front of the crease, but he could not beat the Penguins
goaltender.

Bedard shows off the release, Jarry with the save pic.twitter.com/BKLDkQdpwD — Hockey Realm (@hockey_realm) October 11, 2023

Connor Bedard with his first good scoring chance.
</item 41>

<item 42>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 42>

<item 43>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Turk’s reaction?

Advertisement

“I was like holy f—,” he said, spelling out the whole profanity, “because the only other person that I know who does it quite like that is No. 34 in Toronto
(Auston Matthews), and he still doesn’t bring it in as lateral as Bedzy does.”

After the stationary shooting, they got into some motion drills.

Every time Turk introduced a new formation and shot type for Bedard, his application was “simplistic, and he made it look easy.”

View this post on Instagram A post shared by Tim Turk Hockey (@timturkhockey)

When it was over, both kids said, “This was fun, thanks coach Turky,” and Turk got in his car and drove home.

The next day, he got another call from Landry asking him what he was doing Thursday because the two boys had asked if they could have him back for another
session.

“Well who the f— are these guys?” Turk finally asked him.
</item 43>

<item 44>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— If it feels like Sidney Crosby, Kris Letang and Evgeni Malkin have been in Pittsburgh forever, that's because it had. The trio are set to surpass Yankees
legend Derek Jeter, Mariano Rivera and Jorge Posada as the longest-tenured trio of teammates in North American sports history.

Ready for the Big Three to take the ice for their 18th season together.



Sidney Crosby, Evgeni Malkin and Kris Letang are set to surpass the New York Yankees’ Derek Jeter, Mariano Rivera and Jorge Posada as the longest-tenured trio
of teammates in NHL, MLB, NBA and NFL history. pic.twitter.com/bvcWZOw7kO — Pens Inside Scoop (@PensInsideScoop) October 10, 2023

6:45 p.m. — All the buzz has been about Bedard leading up the matchup, but there's a new Penguin on the roster.
</item 44>

<item 45>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
10:17 p.m. — The Blackhawks are going to their third power play of the night as Matt Nieto is called for hooking just 2:28 into the third.

10:14 p.m. — The third period is a go from Pittsburgh with the Penguins holding onto a one-goal lead.

End of the second period: Penguins 2, Blackhawks 1

9:56 p.m. — Both teams get on the board in the middle frame, but it's the Penguins that remain ahead going into the final period. Bedard looks to be improving
every shift and he picks up his first NHL point on the Donato goal.

9:54 p.m. — An awful line change by the Blackhawks allows Malkin to send Smith on a breakaway, but Mrazek comes up with the stop.

Penguins 2, Blackhawks 1

9:47 p.m. — GOAL!
</item 45>

<item 46>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Erik Karlsson, who won his third Norris Trophy last season, was brought in this summer in a trade with the Sharks. It gives the Penguins a talented duo on the
blue line of Kris Letang and Karlsson, although the two likely won't be paired together.

Actually, this is exactly what we’ve been waiting for. pic.twitter.com/0ZJA4YRs6v — Pittsburgh Penguins (@penguins) October 10, 2023

6:30 p.m. — Connor Bedard is in the house for his NHL debut. It's the most anticipated first game for a rookie since Auston Matthews' debut in 2016.

Blackhawks vs. Penguins start time

Date: Tuesday, Oct. 11

Tuesday, Oct. 11 Time: 8 p.m. ET (7 p.m. CT)

The Blackhawks and Penguins are set for puck drop at 8 p.m. ET, 7 p.m. CT in Chicago.
</item 46>

<item 47>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— GOAL! Perry finds Foligno for the empty net goal. The veteran line gave the Blackhawks the lead, and then the unit ices the game with 1:27 to go.

10:47 p.m. — Penguins head coach Mike Sullivan pulls Jarry for the extra attacker with over two minutes to go.

Blackhawks 3, Penguins 2

10:40 p.m. — GOAL! Chicago takes its first lead of the game thanks to Jason Dickinson. Nick Foligno drops a pass on a zone entry to Perry, who tries to get a
shot on net, but it deflects to the middle and Dickinson snaps it home. The Penguins, who struggled with third period leads last year, have given up three
unanswered and find themselves down one with 4:33 to go.
</item 47>

<item 48>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
9 pick) went first, and Turk could tell he was a talented and hard-working player with a good shot and real promise.

Then, on Connor’s turn, Turk placed him in a position and watched how he got the puck ready to shoot. Only Connor didn’t stickhandle and prepare for the shot,
he just took the puck and shot it right away.

After watching him run through the same number of shots as Nate did, Turk spoke for the first time.

“Timeout one second, just give me a second here, I’m just trying to see what you’re doing so I’m going to have you take an extra few shots so that I can analyze
where the puck’s starting and where it’s finishing,” he said.

Another round later, he knew that this Connor kid, whose last name he still didn’t know and who was then 16, was very different from even his high-end peers
simply by his proficiency in shooting from a “frontal position.”
</item 48>

<item 49>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
For one, he learned to utilize his teammates more. He accumulated plenty of shots on net early in the season, but they were often from distance and more
individualistic chances. Then, he scored his first NHL goal on a wraparound and his next three goals off passes from teammates.

Advertisement

“Look, I’m sure he’ll figure out the one-on-one and how to score that way or create chances that way,” Blackhawks general manager Kyle Davidson said. “There is
also an element of working off your teammates. You are in the NHL, so everyone is a pretty good player and you’re able to create using more than just yourself.
I think he’s figured that out.”

Bedard found more success on the Blackhawks’ last road trip. He left Chicago with five goals in 11 games and returned from Florida with nine goals in 13 games,
scoring twice against the Tampa Bay Lightning and again against the Panthers.

Bedard wasn’t so reliant on teammates on those goals, either.
</item 49>

<item 50>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
It’s happened. That’s their life. They love it so much and they have to be out there. That’s where they feel comfortable and don’t get tired. Hopefully, they go
home and have a nap in the afternoon and not watch ‘Young and the Restless.’ Take it easy and get ready for the next day. It does add up when you put the hours
of the week in.”

Bedard has downplayed his lengthy post-practice sessions. To him, it’s not like he has weights on his ankles. He’s just shooting, nothing too strenuous.

For someone who shoots as much as Bedard does, you’d think he would gladly accept being called a shooter. He doesn’t.

“I say it a lot: I don’t feel like I’m a shooter,” he said. “I’m just trying to make the right play. If the shot’s there, I’ll take it. Obviously, the goalies
are good, you got to hit your spot.”
</item 50>
=== END ITEM id=b1c9d45fc337a318 ===

=== ITEM id=0bfe11f7dd91bab3 ===
Gold sentences and facts:
Sentence 0: Beware of Terry McLaurin (vs. Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.
  Fact 0.0: Beware of Terry McLaurin this week, versus the Dolphins.
  Fact 0.1: Beware of Puka Nacua this week, versus the Browns.
  Fact 0.2: Beware of Tutu Atwell this week, versus the Browns.
  Fact 0.3: Beware of Drake London this week, playing at the Jets.
  Fact 0.4: Beware of Tyler Lockett this week, playing at the Cowboys.
  Fact 0.5: Beware of Jaxon Smith-Njigba this week, playing at the Cowboys.
Sentence 1: HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year.
  Fact 1.0: Harris had over 120 total yards in a Week 17 win in Baltimore last year.
  Fact 1.1: Harris had a touchdown in a Week 17 win in Baltimore last year.

Retrieved items:
<item 1>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential WR busts in Week 14?

It's tough to trust Calvin Ridley (@ Browns), Drake London (vs. Buccaneers), Jordan Addison (@ Raiders), and Jayden Reed (@ Giants) this week. Although Ridley
has produced of late, a brutal matchup against a Browns coverage unit that entered Week 13 allowing the second-fewest fantasy points per game (FPPG) to WRs
raises a red flag, and it's only compounded by the loss of Trevor Lawrence (ankle). London's week-to-week output is way too unpredictable, and with the fantasy
season on the line for some of you, there's no reason to roll London out.

Addison is likely to take a step back with Jefferson back in the fold, while the Packers' receiving situation is too dispersed to put faith into Reed in PPR
formats, though he might get a boost with Christian Watson (hamstring) banged up.
</item 1>

<item 2>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Bowden Jr., alongside A.T. Perry and Keith Kirkwood, are the only healthy Saints WRs right now, and a favorable matchup against Detroit gives each some flex
value in deep leagues. Mingo also draws a plus matchup against a reeling Buccaneers secondary, as a potential negative game script should boost his Week 13
fantasy ceiling.

Tillman is another talented rookie who's getting more run as the season progresses, and although P.J. Walker's struggled under center, Tillman is set up for a
high-usage day against the Rams with Amari Cooper (ribs) banged up.

WEEK 13 DFS:

DK lineup | FD lineup | Best stacks | Best values

Who are the biggest potential WR busts in Week 13?

Beware of Terry McLaurin (vs. Dolphins), Puka Nacua and Tutu Atwell (vs.
</item 2>

<item 3>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the flip side, three of the four quarterbacks Chicago faced found three touchdowns on their limited throws.

Start Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS WR 19th PROJ PTS 15.2 WR RNK 21st YTD Stats REC 21 TAR 26 REYDS 212 TD 1
FPTS/G 12.1 The Commanders may wish to win this one with a big dose of the run game (Brian Robinson is a must-start), but if the Bears put up any offense at
all, McLaurin will be in position to put up nice numbers. He's a low-end WR2 worth the nod over George Pickens, Zay Flowers, Drake London and Jahan Dotson.
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Weekly Breakdown MATCHUP vs JAC BUF -5.5 O/U 48.5 OPP VS WR 11th PROJ PTS 12.6 WR RNK 31st YTD Stats REC 12 TAR 18 REYDS 220 TD 3 FPTS/G 13 Until his role
changes, Davis will be the same kind of Fantasy option every week -- one big play and he's a stud, one or two misses and he's a dud. The Jaguars have done well
against deep-ball receivers but not enough to the point where benching Davis is a good idea. He's toward the top of the flex mix ahead of Tank Dell, Tutu
Atwell, Christian Watson and Jerry Jeudy.

LAST WEEK: Kept playing up to his boom/bust ceiling by catching a long touchdown. That's three straight games with a score. It was also his third game out of
four with four or fewer targets.
</item 4>

<item 5>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 10th PROJ PTS 10.6 WR RNK 31st YTD Stats REC 54 TAR 79 REYDS 575 TD 4 FPTS/G 12.5 Lockett has
seven-plus targets in four of his past five, but he still doesn't have as many as Metcalf's gotten in those games. He also doesn't see nearly as many deep
throws as Metcalf -- literally two targets in his past two games traveled 15 or more Air Yards. Obviously he has potential to always pop off for a big game like
he did against a tough Browns defense five weeks ago, but recent matchups against the Niners, Rams and Ravens weren't kind to Lockett. They were even less kind
to Geno Smith, and this week shouldn't be much different.
</item 5>

<item 6>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Bills

65 Tutu Atwell, Rams @ Ravens

66 Treylon Burks, Titans @ Dolphins

67 Josh Reynolds, Lions @ Bears

68 Nick Westbrook-Ikhine, Titans @ Dolphins

69 Cedric Tillman, Browns vs. Jaguars

70 Marquez Valdes-Scantling, Chiefs vs. Bills

71 Dontayvion Wicks, Packers @ Giants

72 Hunter Renfrow, Raiders vs. Vikings

73 Darnell Mooney, Bears vs. Lions

74 Skyy Moore, Chiefs vs. Bills

75 Kalif Raymond, Lions @ Bears

76 A.T. Perry, Saints vs. Panthers

77 Cedrick Wilson, Dolphins vs. Titans

78 Trenton Irwin, Bengals vs. Colts

79 Michael Gallup, Cowboys vs.
</item 6>

<item 7>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least all of them went to his wideouts. All were in the red zone, none were on deep throws and none were on WR screens.

Sit Him (Lineup Decisions) Projections powered by Sportsline Najee Harris RB PIT Pittsburgh • #22

Age: 25 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs BAL BAL -4 O/U 38 OPP VS RB 12th PROJ PTS 9.3 RB RNK 30th YTD Stats RUYDS 210 REC 4 REYDS 34 TD 0
FPTS/G 7.1 The Steelers' lack of scoring opportunities really crushes the upside for Harris, who admittedly has improved this season and could be on his way to
being a reliable RB2. The Ravens have been pretty good against the run but they can give up some numbers. Harris is a volume-based back who is at best a low-end
starting option given everything that's going on.
</item 7>

<item 8>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.

McLaurin's averaging just 42 receiving yards over his past three games, and a heavy dosage of Jalen Ramsey hinders his Week 13 fantasy value.

Nacua and Atwell see a Cleveland coverage unit who entered Week 12 allowing the fewest fantasy points per game to WRs. Heck, even Cooper Kupp, who hasn't looked
like himself of late, could underperform again.

We can't trust London and the rest of the Falcons passing offense against a well-rounded Jets defense. To make matters worse, he'll be matched up against Sauce
Gardner for most of the day, too.

Lockett and Smith-Njigba might be better suited for the bench in PPR leagues against a lockdown Dallas coverage unit.
</item 8>

<item 9>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Even concepts like utilizing Jalen Hurts' mobility on downfield throws and targets to D'Andre Swift are things the Niners have dealt with successfully this
season. Not that you'll sit any of your usual Eagles starters in Fantasy, but some might underwhelm given the circumstances. P.S.: They play at Dallas next
week.

STARTS: Jalen Hurts, Christian McCaffrey, A.J. Brown, DeVonta Smith, Brock Purdy, Brandon Aiyuk, Deebo Samuel, George Kittle, D'Andre Swift, 49ers DST

Jalen Hurts, Christian McCaffrey, A.J.
</item 9>

<item 10>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Pair that with the expectation that he'll line up across from DaRon Bland for much of the game (Bland usually plays right cornerback; Metcalf primarily lines up
at left wide receiver) and it could be a rough one. Bland has allowed one touchdown and has allowed just a 52.5% catch rate with 13 passes defensed and seven
interceptions. It's the volume you're banking on this week -- I'd start Metcalf in PPR over Diontae Johnson and Cooper Kupp, but I like the upside for Rashee
Rice and Courtland Sutton just a little bit more.

Flex Starter (Lineup Decisions) Projections powered by Sportsline Tyler Lockett WR SEA Seattle • #16

Age: 31 • Experience: 9 yrs.
</item 10>

<item 11>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 11>

<item 12>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
They were even less kind to Geno Smith, and this week shouldn't be much different. Tack on the tough assignment in Stephon Gilmore in coverage (unless the
Cowboys use shadow coverage for the first time this year), and Lockett could be in for another lean night. I don't mind him as a WR3 over Drake London, DeAndre
Hopkins and Diontae Johnson, but I'd start Rashee Rice and Adam Thielen over Lockett.

Sit Him (Lineup Decisions) Projections powered by Sportsline Jake Ferguson TE DAL Dallas • #87

Age: 24 • Experience: 2 yrs.
</item 12>

<item 13>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 13>

<item 14>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 14>

<item 15>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
Because we splurged for McCaffrey and Kamara, we had to sacrifice salary elsewhere, and London is a high-upside option even in an offense that doesn't throw the
ball with efficiency.

WR Jaxon Smith-Njigba, Seahawks @ 49ers ($4,100). Is a takeover underway in the Seahawks offense? Smith-Njigba is coming off a game against the Cowboys that saw
him collect seven-of-11 targets for 62 yards. It's a step up from a disappointing outing against San Francisco, but Tyler Lockett's involvement seems to be
lessening as of late with the rookie stepping up. It wasn't a great day for the Seattle offense on Thanksgiving against the Niners, but the Seahawks should put
up a better effort against a divisional opponent they know so well. If they're trailing, JSN should see action with a negative game script.
</item 15>

<item 16>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Brissett didn't throw downfield as much as Walker, contributing to a higher ADOT for Walker, but not by much (8.8 to 8.3). Both quarterbacks were right around
6.5 yards per pass attempt.

49ERS: Oh yeah, the matchup isn't great. No team allows fewer yards per catch to wideouts than the Niners (9.7), and they're thrown at a ton by offenses that
have to play from behind. Only three receivers -- Puka Nacua (20 targets), Marquise Brown (10 targets) and Michael Wilson (two touchdowns) have had more than 15
PPR points against them. And only one other receiver -- Tutu Atwell (nine target) -- had more than 12 PPR points.
</item 16>

<item 17>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.
</item 17>

<item 18>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST

Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST SITS: Sam Howell (low-end
starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week replacement), Commanders DST

Sam Howell (low-end starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week
replacement), Commanders DST SLEEPERS: Curtis Samuel (especially PPR), Jahan Dotson (non-PPR only)

Start 'em 10.0-7.00 6.99 - 5.01 5.
</item 18>

<item 19>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He has more upside than Jerry Jeudy, Jahan Dotson and Tutu Atwell but I'd rather take a shot on Drake London, Terry McLaurin or Chris Godwin.

LAST WEEK: Over 19 snaps without Justin Jefferson last week, Osborn ran 18 routes and caught both of his two targets for 30 total yards. The receiving average
was delightful, but the small-sample target share (12.5%) was saddening. What's funny is that Osborn's target share was much higher when Jefferson was playing
(23.3%) but his receiving average tanked (6.3).

FILM: Osborn didn't beat coverage with his speed last week and his burst and agility were just OK. His 22-yard catch came on a Kirk Cousins scramble where he
waited in a spot and Cousins hit him, then Osborn ran inside instead of outside where he might have been able to score.
</item 19>

<item 20>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Tack on the ascending secondary the Falcons seem to have and the Commanders' diffusion of targets and McLaurin is at best a low-end flex. I'd rather roll the
dice on Drake London, Tyler Lockett or Garrett Wilson.

LAST WEEK: Hopeful that his target edge over his teammates would keep growing, and even in a favorable game script where the Commanders were trailing by three
scores before halftime, McLaurin only saw five targets against the Bears' heavy zone coverage.

FILM: On the 10 third-down routes McLaurin ran last week, he drew extra coverage on four of them, got open on two of them (Howell threw elsewhere) and was a
decoy on Curtis Samuel's touchdown. What happened? Sam Howell was getting rid of the ball quickly and to shorter distances as the Bears were getting some
pressure on him. By the end of the game McLaurin's body language seemed dispirited.
</item 20>

<item 21>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
By the end of the game McLaurin's body language seemed dispirited.

COMMANDERS: Throw to their wideouts on 58.7% of their targets, which is below league average (but not as bad as, say, the Falcons at 36.4%). What makes that
number worse is that McLaurin has just one more target (31) than Jahan Dotson (30), who has three more targets than Curtis Samuel (27). They're all nearly even
after five games and each of them has had a turn leading in targets at least once.

FALCONS: Are expected to play a heavy dose of zone coverage, if only because the Commanders have struggled to be explosive against zone coverage this year.
Atlanta also gave a full dose of snaps last week to outside cornerback Jeff Okudah, who wasn't flawless but does offer another sizable body to help against
stronger perimeter receivers. Between Okudah and A.J.
</item 21>

<item 22>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PAST THREE WEEKS: McLaurin has led all Commanders wide receivers in targets and all Commanders players in receptions. He's the No. 1 guy, and his big Week 4
game (8-86-0 with an offensive fumble recovery for a touchdown) would have been even bigger if Sam Howell had better ball placement on a deep pass McLaurin was
open for.

BEARS: Are expected to play again without top outside cornerback Jaylon Johnson and safety Eddie Jackson. Not that those two absences led to a bonanza from
Broncos receivers; Russell Wilson spread the ball around. But if McLaurin remains the target leader then he'll have the best shot at a big game. Rookies Tyrique
Stevenson and Terell Smith figure to line up across from McLaurin the most.
</item 22>

<item 23>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 23>

<item 24>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Between Okudah and A.J. Terrell there isn't a matchup that's great for McLaurin.

McLAURIN: Has an impressive 91.3% catch rate versus zone coverage this year, but is averaging just 9.3 yards per catch with it on a sub-9.0 ADOT. His target
share is below 20% regardless of the kind of coverage he sees.

Start Him (Lineup Decisions) Projections powered by Sportsline Logan Thomas TE WAS Washington • #82

Age: 32 • Experience: 9 yrs. Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS TE 31st PROJ PTS 8.4 TE RNK 11th YTD Stats REC 18 TAR 25 REYDS 183 TD 2
FPTS/G 11.6 Streaming Thomas is worth the risk considering how the Falcons have let up numbers to tight ends.
</item 24>

<item 25>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Chiefs

31 Diontae Johnson, Steelers vs. Cardinals

32 Tee Higgins, Bengals @ Jaguars

33 George Pickens, Steelers vs. Cardinals

34 Jerry Jeudy, Broncos @ Texans

35 Tyler Lockett, Seahawks @ Cowboys

36 Drake London, Falcons @ Jets

37 Noah Brown, Texans vs. Broncos

38 Josh Downs, Colts @ Titans

39 Brandin Cooks, Cowboys vs. Seahawks

40 DeVante Parker, Patriots vs. Chargers

41 Jahan Dotson, Commanders vs. Dolphins

42 Jayden Reed, Packers vs. Chiefs

43 Amari Cooper, Browns @ Rams

44 Jaxon Smith-Njigba, Seahawks @ Cowboys

45 Curtis Samuel, Commanders vs. Dolphins

46 Elijah Moore, Browns @ Rams

47 Rondale Moore, Cardinals @ Steelers

48 Tyler Boyd, Bengals vs. Jaguars

49 Robert Woods, Texans vs.
</item 25>

<item 26>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 26>

<item 27>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The Commanders offensive line has had some ugly showings lately against the Cowboys (four sacks, 49% pass rush pressure rate allowed), the Giants (four sacks,
32.7% pressure rate allowed) and the Seahawks (three sacks, 46.8% pressure rate allowed). Somehow these results are better than what they were allowing through
much of the first two months of the year. Sam Howell getting the ball out faster has been a positive in this regard but a negative for his numbers in three of
his past four games (the one game he did really well in was sort of fluky). The matchup is a good test for the Dolphins pass rush post-Phillips, one that it
should pass without giving up much in the way of deep completions.

STARTS: Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2),
</item 27>

<item 28>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 28>

<item 29>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS WR 6th PROJ PTS 12.1 WR RNK 37th YTD Stats REC 25 TAR 31 REYDS 261 TD 1
FPTS/G 11.4 The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2 and Washington's 34-31 OT loss to Philadelphia
in Week 4. This matchup has an over/under of 42.5 points, tied for fourth-lowest of the week.
</item 29>

<item 30>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Jaguars

49 Robert Woods, Texans vs. Broncos

50 Jonathan Mingo, Panthers @ Buccaneers

51 Jameson Williams, Lions @ Saints

52 Zay Jones, Jaguars vs. Bengals

53 Tutu Atwell, Rams vs. Browns

54 Greg Dortch, Cardinals @ Steelers

55 Jalen Guyton, Chargers @ Patriots

56 Lynn Bowden Jr., Saints vs. Lions

57 Quentin Johnston, Chargers @ Patriots

58 Justin Watson, Chiefs @ Packers

59 Treylon Burks, Titans vs. Colts

60 Josh Reynolds, Lions @ Saints

61 Keith Kirkwood, Saints vs. Lions

62 Cedric Tillman, Browns @ Rams

63 Trey Palmer, Buccaneers vs. Panthers

64 A.T. Perry, Saints vs. Lions

65 Michael Gallup, Cowboys vs. Seahawks

66 JuJu Smith-Schuster, Patriots vs.
</item 30>
=== END ITEM id=0bfe11f7dd91bab3 ===
