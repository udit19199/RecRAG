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

=== ITEM id=04a5d54c7001e36e ===
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
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Composure and step through congestion

AFL player comparison: Adam Cerra (Carlton)

George Stevens

State: Victoria Country

State League Club: GWV Rebels

Community Club: South Warrnambool

Date of Birth: 14/04/05

Height: 189cm

Advertisement

A strong-bodied defender who also plays as an inside midfielder to great effect. He reads the cues to intercept well when playing across half back, is a good
decision-maker and has a penetrating kick. Had an outstanding year which included captaining the Australia under-18 team during the AFL Academy’s matches
against SANFL and VFL sides. Won All Australian honours at the National Championships when representing Victoria Country and was named captain of the Coates
Talent League Team of the Year after averaging 28.8 disposals and five tackles across 13 matches for the GWV Rebels, also winning the club’s best and fairest
award.
</item 1>

<item 2>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— GOAL! Perry finds Foligno for the empty net goal. The veteran line gave the Blackhawks the lead, and then the unit ices the game with 1:27 to go.

10:47 p.m. — Penguins head coach Mike Sullivan pulls Jarry for the extra attacker with over two minutes to go.

Blackhawks 3, Penguins 2

10:40 p.m. — GOAL! Chicago takes its first lead of the game thanks to Jason Dickinson. Nick Foligno drops a pass on a zone entry to Perry, who tries to get a
shot on net, but it deflects to the middle and Dickinson snaps it home. The Penguins, who struggled with third period leads last year, have given up three
unanswered and find themselves down one with 4:33 to go.
</item 2>

<item 3>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Rust gets the Penguins their first of the season 🚨 pic.twitter.com/RKNyaJT0LD — B/R Open Ice (@BR_OpenIce) October 11, 2023

8:28 p.m. — It's a strong showing from the top PP unit for the Blackhawks, but Bedard and Co. can't capitalize on the man advantage. Chicago had four shots on
goal, including a Bedard one-timer that was stopped by the pad of Mrazek.

8:26 p.m. — Penguins forward Lars Eller is called for goalie interference, as he clips the leg of Petr Mrazek in the crease. The Blackhawks head to the first
power play of the game about five minutes into the period.

8:23 p.m. — Through the first four minutes, it's been a back-and-forth affair. Neither team have been able to sustain much pressure. Most of Chicago's shot
attempts have been from the outside, making it easy on Tristan Jarry.
</item 3>

<item 4>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
9 pick) went first, and Turk could tell he was a talented and hard-working player with a good shot and real promise.

Then, on Connor’s turn, Turk placed him in a position and watched how he got the puck ready to shoot. Only Connor didn’t stickhandle and prepare for the shot,
he just took the puck and shot it right away.

After watching him run through the same number of shots as Nate did, Turk spoke for the first time.

“Timeout one second, just give me a second here, I’m just trying to see what you’re doing so I’m going to have you take an extra few shots so that I can analyze
where the puck’s starting and where it’s finishing,” he said.

Another round later, he knew that this Connor kid, whose last name he still didn’t know and who was then 16, was very different from even his high-end peers
simply by his proficiency in shooting from a “frontal position.”
</item 4>

<item 5>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
And for Poyer, there was value in being someone else.

What his teammates didn’t know was that Poyer had not put alcohol to his lips in about three months, since he downed a shot of tequila on March 13 in the Puerto
Vallarta airport bar during a 12-hour flight delay.

Poyer spent his childhood in Astoria, Ore., a picturesque town on the mouth of the Columbia River not far from the Pacific Ocean. His mother and stepfather
worked at a juvenile detention center and raised him with a strict hand. When he was no longer under their watch, unsupervised as a freshman at Oregon State,
Poyer drank. And drank and drank.

It never seemed to affect his play in college, where he was a consensus All-American. And despite his continued drinking, he made steady progress in his early
NFL years.
</item 5>

<item 6>
Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' (Fortune, 2023-10-13T22:26:03+00:00)
How the heck did we do that?

You’ve said the King-Riggs match was about social change, women standing up for themselves in all areas.

It was really about men, too. Because men started to shift a little. Obama was 12 years old when he saw the King-Riggs match. He said it affected him a lot.
Guys are much better thinking about their daughters than they used to be. All these things add up.

You’re part of ownership groups for pro sports. How did you get involved in women’s pro hockey, which will have teams in Boston, New York, Minnesota, Toronto,
Ottawa and Montreal?

The PWHL, it’s really exciting. It took five years. Plus, it took all those years of the other leagues, everyone trying. (U.S. Olympic gold medalist) Kendall
Coyne said, ‘can you help us?’ We need to have a league where the very best players will play.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 7>

<item 8>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“He really likes to pull pucks right into his feet. I really like his mechanics with his hands off his body and whippy stick — but I feel this habit makes it
easier for elite level ‘D’ to get stick on puck or in the lanes to block shots. Finding space and time to let that shot go and get clean looks, I think, will be
an adjustment.”

For goalies, it’s the angle change that gets them.

Moose Jaw Warriors goalie Jackson Unger, whom Bedard scored on last year, said the toe-drag release, in particular, is a challenge.

“He changes the angle so quickly that, as a goalie, you have to adjust to it, but when he does it so fast, it’s easier said than done,” Unger said. “It’s a lot
of different angles he can give you.”
</item 8>

<item 9>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Showcased his athletic prowess at the national Draft Combine where he produced the quickest time on the agility test (8.16 seconds) and a top ten finish on the
20m sprint (2.92 seconds). Is a member of the West Coast Eagles’ Next Generation Academy meaning they can match any bid after selection 40 in this year’s Draft.

Advertisement

Weapon: Speed and innate goal sense

AFL player comparison: Charlie Cameron (Brisbane Lions)

Jordan Croft

State: Victoria Metro

State League Club: Calder Cannons

Community Club: Maribyrnong Park

Date of Birth: 12/05/05

Height: 200cm

A tall forward who is lightly built but extremely athletic. He uses his mobility and height to create opportunities as a marking forward, with his speed and
precise kicking key features of his game. He only fully focused on a potential AFL career last year after previously playing volleyball.
</item 9>

<item 10>
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
</item 10>

<item 11>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 11>

<item 12>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Uses his dazzling speed to break the lines and kicks the ball with pinpoint accuracy on his preferred left foot. Averaged 32.8 disposals and 5.8 clearances
across four matches for the Allies at the National Championships to earn All Australian honours. Was equally dominant for the Tasmania Devils in the Coates
Talent League having more than 30 disposals in seven of his nine matches and also kicked 12 goals. Showed he could match it against more experienced opponents
by having 20 disposals and three clearances in Australia’s win over Carlton’s VFL team.
</item 12>

<item 13>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
A competitive type who has a good natural leap and is very strong overhead. He had an excellent year at under-18 level for Central District where he averaged
21.8 disposals and 5.9 marks across nine matches. He was also one of South Australia’s better players at the National Championships. Is eligible to join
Hawthorn under the father-son rule as the son of Luke McCabe who played 138 games for the Hawks from 1995 to 2004.

Weapon: Overhead marking

AFL player comparison: Jordan Ridley (Essendon)

Advertisement

Colby McKercher

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 12/04/05

Height: 180cm

Had an outstanding year which included winning the Coates Talent League’s Morrish Medal and finishing runner-up in the Larke Medal at the National
Championships.
</item 13>

<item 14>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.
</item 14>

<item 15>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 15>

<item 16>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Played all four games for Victoria Metro at the National Championships and was his team’s best player in the loss to the Allies, while also shining with 22
disposals in the win over Western Australia. Across his 16 matches with the Eastern Ranges in the Coates Talent League he averaged 19.4 disposals to be one of
their most consistent players during their run to the grand final. Recorded a slick time of 2.92 seconds on the 20m sprint during the national Draft Combine
highlighting his dazzling acceleration and backed it up by recording 6:28 for the 2km time trial which demonstrated a combination of elite speed and endurance
which are essential to play on the wing.

Weapon: Elite speed and ball use

AFL player comparison: Nick Martin (Essendon)
</item 16>

<item 17>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Petr Mrazek with a nifty glove stop on Reilly Smith#Blackhawks | #NHLFaceoff pic.twitter.com/AkE9PEgRrZ — Hockey Daily 365 l NHL Highlights & News
(@HockeyDaily365) October 11, 2023

9:21 p.m. — Pittsburgh kills off the Carter minor. Chicago made it fairly easy on that power play, putting just one shot on net and spending a majority of the
man advantage passing around the perimeter.

9:19 p.m. — Jeff Carter gives the Blackhawks an early power play, as the Penguins veteran is called for holding the stick of Korchinski. Chicago gets its second
crack on the man advantage.

9:15 p.m. — Puck is down for the second period in Pittsburgh.

End of the first period: Penguins 1, Blackhawks 0

8:56 p.m. — The first 20 minutes features tons of shot attempts, but Rust's tally is the only one on the scoreboard.
</item 17>

<item 18>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
But just because injuries might take some of the juice out of these games doesn't mean that SuperDraft isn't giving us plenty of reasons to care. There are some
snazzy looking props out there for us to make our bottom line look at lot better than it did earlier this morning.

SuperDraft users can win 3X the amount wagered if their two-legged player prop parlay hits. If all four of the props below hit, users will win 10x their entry.
In other words, a $20 entry fee will yield a $200 payout. the more legs you add to your player prop parlay, the larger your potential payout becomes, but as the
long odds indicate, completing a clean sweep is challenging.

Below, we'll break down four of our favorite player props for Tuesday night's NHL slate. Users don't necessarily have to parlay all four picks together, though.
</item 18>

<item 19>
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
</item 19>

<item 20>
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
</item 20>

<item 21>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was an outstanding player for Victoria Country at the National Championships to earn All Australian honours, which included a match-winning performance against
Victoria Metro in the final game where he amassed 22 disposals, 10 marks and four goals. Was also dominant for the Gippsland Power to win their best and fairest
and earn selection in the Coates Talent League Team of the Year. Finished the year with 42 goals across 20 matches in all competitions. His natural leap was
highlighted at the national Draft Combine with his running vertical jump of 97cm ranked second overall. Hails from a talented football family, with his brother
Xavier now at Essendon and sister Yasmin playing for Port Adelaide.
</item 21>

<item 22>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Aerial skill and goal sense

AFL player comparison: Isaac Heeney (Sydney Swans)

Charlie Edwards

State: Victoria Country

State League Club: Sandringham Dragons

Community Club: Sandhurst

Date of Birth: 18/05/05

Height: 191cm

Medium defender and midfielder who emerged as the year progressed and is now considered a possible Draft bolter given his rapid progression for the Sandringham
Dragons in their road to the Coates Talent League premiership. Showed real class with his athleticism and ball use once moved into the midfield with his
29-disposal performance against the Dandenong Stingrays in Round 13 an eye-catcher. Is also a talented cricketer and has spent only one year in the AFL Talent
Pathway system after playing underaged football at Sandhurst before moving to Melbourne Grammar.
</item 22>

<item 23>
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
</item 23>

<item 24>
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
</item 24>

<item 25>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
But they also haven’t really beaten anyone. They were down 14 points to Carolina. Do you think they are coming back from 13 down against Philly? In my opinion,
no.”

Advertisement

• Dallas Cowboys: Is this team any better than previous Dallas teams that fell short?

What I think: It’s a tough time to evaluate the Cowboys fairly because they were so bad against San Francisco on a big stage in their most recent game. I just
don’t see where Dallas is better or going to get considerably better, especially on offense. And every time they enjoy success, they seem to enjoy it a little
too much. That is a reflection of their owner.

Exec comment: “Run defense and consistency on defense, and then offensively, what is their identity? They just are not very good. Their best games are when they
play great defense and get turnovers, but that is not going to happen enough, especially against the good teams.”

• Detroit Lions: Have they fixed their defense?
</item 25>

<item 26>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
His 3.39 mark across 193 2/3 innings this past season wasn’t quite so dominant but is still high-end production.

Burnes struck out just under 26% of opposing hitters. While his lowest mark since his rookie season, that’s still an above-average figure.

Adames is coming off a less impressive year. The 28-year-old hit .217/.310/.407 through 638 trips to the plate. That’s slightly below-average offense, although
he still connected on 24 home runs.

Adames is an excellent defensive shortstop with plus power in an offseason where there are essentially no free agent shortstops of note. Even with a mediocre
on-base percentage, he’d generate plenty of interest.

4. Shane Bieber, SP, Guardians

Develop high-end starting pitching, trade a top starter as they become more expensive, replace him with talented younger arms. The Guardians have trodden this
path with Corey Kluber, Mike Clevinger and Carlos Carrasco.
</item 26>

<item 27>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Jason Dickinson buries the crazy carom to give Chicago the late lead in Pittsburgh!#Blackhawks pic.twitter.com/rSxEzSnylC — Hockey Daily 365 l NHL Highlights &
News (@HockeyDaily365) October 11, 2023

10:36 p.m. — Mrazek records another sensational save, this time on Guentzel. Karlsson makes an excellent backhand pass down to Guentzel at the dot for a
one-timer, but Mrazek pushes over and snags the shot. He's been the best player on either side tonight.

Penguins 2, Blackhawks 2

10:31 p.m. — GOAL! We have a tied game in Pittsburgh as Guttman gets the equalizer for Chicago. He found a soft spot in the Penguins defense in the slot, gets a
pass from Jones and makes no mistake, beating Jarry by the glove. It's 2-2 with 9:55 left in regulation.
</item 27>

<item 28>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
20. Brandon Drury, 2B, Angels

The Halos signed Drury to a two-year free agent deal last winter. The righty-hitting infielder turned in a nice season, connecting on 26 home runs with a
.262/.306/.497 line across 523 plate appearances. Next year’s $8.5M salary looks like strong value, particularly in an offseason where the supply of
up-the-middle players is thin.

The Angels probably aren’t kicking off a rebuild. They could do a shorter-term retool if Ohtani walks, or they could simply look to move Drury for pitching
while turning second base to Luis Rengifo and/or an external acquisition.

21. Brendan Donovan, 2B, Cardinals

Donovan is a longer shot trade possibility if the Cards move a controllable hitter for starting pitching. He’s more valuable than Carlson and less likely to be
on the move.

At the same time, he’d net a clearer rotation upgrade if St.
</item 28>

<item 29>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
You just never how quickly.

“It’s a big adjustment, especially when you’re playing against men and being focused on by the other team every single night. I’m excited for him, but I don’t
necessarily feel like it’s a huge surprise how quickly he’s started to really find his groove at the NHL level. But it’s exciting to watch.”

(Top photo: Ethan Miller / Getty Images)
</item 29>

<item 30>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
A very clean ball handler with a dazzling burst of speed, he was a consistent performer across all four matches for the Allies, averaging 16.5 disposals at 80
percent efficiency. His speed and power really shone through at the national Draft Combine where he was ranked second in the 20m sprint (2.92 seconds) and
standing vertical jump (78cm) as well as being third in the running vertical jump (95cm) which confirmed his elite athletic profile.

Weapon: Speed and ball use

AFL player comparison: Dan Houston (Port Adelaide)

Will Green

State: Victoria Metro

State League Club: Northern Knights

Community Club: Northcote

Date of Birth: 08/09/05

Height: 204cm

Developing tall who impressed throughout the year with his ruck craft, marking and mobility when playing for the Northern Knights, Victoria Metro and the
Australia under-18 team.
</item 30>

<item 31>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
New York has subsequently walked back some of their statements about a possible retool. They’re still aiming to compete in ’24 even if they’re signaling they
won’t blow away the top of the free agent market.

First-year president of baseball operations David Stearns has twice publicly said he anticipates Alonso being the Opening Day first baseman in Queens. That
won’t stop teams from trying, but an offseason deal looks decidedly unlikely.

Others of note

A’s: Seth Brown

Braves: Vaughn Grissom

Brewers: Adrian Houser, Devin Williams, Brandon Woodruff

Cardinals: Alec Burleson, Tommy Edman, Nolan Gorman, Tyler O’Neill

Giants: J.D.
</item 31>

<item 32>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Erik Karlsson, who won his third Norris Trophy last season, was brought in this summer in a trade with the Sharks. It gives the Penguins a talented duo on the
blue line of Kris Letang and Karlsson, although the two likely won't be paired together.

Actually, this is exactly what we’ve been waiting for. pic.twitter.com/0ZJA4YRs6v — Pittsburgh Penguins (@penguins) October 10, 2023

6:30 p.m. — Connor Bedard is in the house for his NHL debut. It's the most anticipated first game for a rookie since Auston Matthews' debut in 2016.

Blackhawks vs. Penguins start time

Date: Tuesday, Oct. 11

Tuesday, Oct. 11 Time: 8 p.m. ET (7 p.m. CT)

The Blackhawks and Penguins are set for puck drop at 8 p.m. ET, 7 p.m. CT in Chicago.
</item 32>

<item 33>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— Chicago kills off the delay of game penalty, so it's back to even strength. For all the star power on the top unit, the Pittsburgh power play has not been as
crisp as expected.

Penguins 2, Blackhawks 0

9:42 p.m. — GOAL! Crosby's goal stands, as the official determine the play was onside. A great cycle from Pittsburgh results in the captain putting home his
first of the year with 8:04 to go. With the unsuccessful challenge, the Blackhawks are hit with a delay of game penalty, so the Penguins get the goal and a
power play.

JUST CROSBY DOING HIS THING 💥 pic.twitter.com/AN0gLjd2lY — B/R Open Ice (@BR_OpenIce) October 11, 2023

9:39 p.m. — The Penguins may have doubled their lead, as Crosby sends a loose puck into the wide open net for the score off a give-and-go with Guentzel.
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
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“It’s a lot of different angles he can give you.”

Sabres first-round pick Zach Benson, who has played on teams with and against Bedard and skates with him in the summers, says it’s Bedard’s ability to set up
the shot with his agility pre-shot that stands out to him.

“He can move left to right like no one I’ve seen before,” Benson said.

“It’s just trying to get out, face him, and hope it hits ya,” said Scott Ratzlaff, Seattle Thunderbirds goalie and Buffalo Sabres prospect, with a smile. “He
can shoot from anywhere and he’s lethal from anywhere. You’ve just always got to be ready just in case he shoots it. And then he’s got a really good toe drag
release, so it’s watching for that and making sure you’re lined up.”

A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting.
</item 35>

<item 36>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“So many players pull the puck in but can’t get the puck to release from under their body. His footwork is so underrated in that aspect. If you watch his front
leg, at times it’s literally in the air at release — most have their back leg in the air at release. When he transfers his weight, he actually clears space for
his hands to get tighter because he’s not afraid to actually lunge into the shot.”

A fourth shooting coach, who hasn’t worked with Bedard and also requested anonymity because he works for another NHL club, expressed a little more hesitancy.

Advertisement

“I’m in the minority I’m sure, but I am very curious to see how well his trademark shooting style works at this level,” the coach said. “He really likes to pull
pucks right into his feet.
</item 36>

<item 37>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
They are, at least, likely to qualify for the play-offs, and therefore will have two must-win games in March to secure their first major tournament appearance
since 2000 — the summer when Haaland was born. But there’s been little in recent performances to think Norway will breeze through those play-offs.

Juve’s individual record will soon be surpassed, but captaining his side to a bronze medal at the Olympics may stand as his nation’s greatest achievement for
much longer.

GO DEEPER Erling Haaland is phenomenal - so why hasn’t he made Manchester City better?

(Top photo: Sebastian Widmann – UEFA/UEFA via Getty Images)
</item 37>

<item 38>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
Minnesota has not won since Nov. 7 against the Islanders and is winless in seven straight.

Woods, who had worked on the Capitals’ bench with Evason briefly as an assistant in the early 2010s, is now on the open market after more than six seasons with
the Wild. It’s unclear whether the Wild will opt to fill his vacancy.

Hynes, who was let go by the Predators after last season, is technically in the final season of his contract with Nashville. That means the Predators had to
grant permission to their divisional rival to speak to their former head coach and that Hynes will be paid by both Minnesota and Nashville this season.

Now behind the bench of his third NHL franchise, he faces an uphill climb to get Minnesota back in the playoff picture. Hynes will suit up behind an NHL bench
for a ninth consecutive season, as he was fired by the Devils and hired by Nashville midway through the 2019-20 campaign.
</item 38>

<item 39>
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
</item 39>

<item 40>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— If it feels like Sidney Crosby, Kris Letang and Evgeni Malkin have been in Pittsburgh forever, that's because it had. The trio are set to surpass Yankees
legend Derek Jeter, Mariano Rivera and Jorge Posada as the longest-tenured trio of teammates in North American sports history.

Ready for the Big Three to take the ice for their 18th season together.



Sidney Crosby, Evgeni Malkin and Kris Letang are set to surpass the New York Yankees’ Derek Jeter, Mariano Rivera and Jorge Posada as the longest-tenured trio
of teammates in NHL, MLB, NBA and NFL history. pic.twitter.com/bvcWZOw7kO — Pens Inside Scoop (@PensInsideScoop) October 10, 2023

6:45 p.m. — All the buzz has been about Bedard leading up the matchup, but there's a new Penguin on the roster.
</item 40>

<item 41>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Averaged 17.1 disposals at 76 percent efficiency and kicked seven goals across 12 matches in the Coates Talent League to announce himself as a Draft prospect.
His 6:24 in the 2km time trial at the state Draft Combine in Victoria was impressive for a player of his type.

Weapon: Rebound and ball use

AFL player comparison: Bailey Dale (Western Bulldogs)

Mitchell Edwards

State: Western Australia

State League Club: Peel Thunder

Community Club: South Mandurah

Date of Birth: 02/06/05

Height: 206cm

Advertisement

An exciting ruckman still growing into his body. Has the height and natural leap to give his team a distinct advantage at stoppages. Has continued to develop
this year and shown glimpses of his aerial ability at state under-18 level and took several exciting marks during the National Championships.
</item 41>

<item 42>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
veterans. The numbers stack up pretty evenly across various distance cutoffs. Based on these numbers, there’s no statistical indication rookies are less
effective in the postseason.

“Playoff teams have a different aura, different feel in the locker room,” a veteran evaluator said. “Percentages are one thing, but who is the guy who trots out
there to try the game-winner? It’s a different pressure shooting free throws up by 20, down by 20, than in that type of situation.”

Advertisement

Fans might recall rookie kickers missing playoff kicks: Rodrigo Blankenship with the Colts in 2020, Harrison Butker with the Chiefs in 2017, Nate Kaeding with
the Chargers in 2005. But the NFL has seen plenty of rookies kick effectively in the postseason as well. Evan McPherson made all 14 postseason tries with the
2021 Super Bowl Bengals. The 2017 Eagles won a Super Bowl with rookie Jake Elliott performing well.
</item 42>

<item 43>
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
</item 43>

<item 44>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The 59-year-old Evason began his coaching career with the WHL’s Calgary Hitmen as an assistant in 1998-99 after finishing out his playing career in Germany,
working his way up the junior hockey ranks over the next handful of seasons. His first NHL job came courtesy of an assistant role with the Capitals immediately
after the 2004-05 lockout, coaching Alex Ovechkin in his first seven NHL seasons.

After departing Washington in 2012, Evason spent the next several seasons working in the Predators organization as the head coach of their AHL affiliate, the
Milwaukee Admirals. There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor
Arvidsson, and Kevin Fiala, among many others. After joining the Wild, he earned a promotion to head coach within two seasons when the Wild fired Bruce Boudreau
late in the 2019-20 campaign.
</item 44>

<item 45>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
While almost all players point their toes at the net, set the puck at a right angle, load their shot and “drag it in and then snap it off,” Turk says, on a line
of 40 to 50 degrees behind their body or at least level with their heels, Bedard brought the puck in “completely lateral.”

Instead of drawing the puck in at a 45-degree angle, he drew it at 180 degrees “right across to the plane of his spine.” Sometimes it even came in at a negative
angle. It also happened in a smaller space than it does for other shooters. That lateral inward pull was “condensed — it’s compacted.”

Turk says that Vladimir Tarasenko, whom Turk calls one of the NHL’s quickest shooters, takes 18 inches to pull the puck in. Bedard did it all in 12 square
inches — a one-foot box.

Turk’s reaction?
</item 45>

<item 46>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Shared the role of Victoria Country co-captain with Harley Reid and performed consistently. Had an exciting year with the Dandenong Stingrays averaging 19
disposals and 4.8 tackles to earn selection in the Coates Talent League Team of the Year. Recorded a top 10 finish in the 20m sprint at the national Draft
Combine with a slick 2.99 seconds. A dual sports athlete who was a member of Australia’s under-19 cricket program.

Weapon: Speed and endurance

AFL player comparison: Jordan Dawson (Adelaide Crows)

Zane Duursma

State: Victoria Country

State League Club: Gippsland Power

Community Club: Foster

Date of Birth: 28/08/05

Height: 189cm

A mercurial forward or midfielder who reads the play exceptionally well, is strong overhead and deadly accurate around the goals.
</item 46>

<item 47>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 47>

<item 48>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
It’s happened. That’s their life. They love it so much and they have to be out there. That’s where they feel comfortable and don’t get tired. Hopefully, they go
home and have a nap in the afternoon and not watch ‘Young and the Restless.’ Take it easy and get ready for the next day. It does add up when you put the hours
of the week in.”

Bedard has downplayed his lengthy post-practice sessions. To him, it’s not like he has weights on his ankles. He’s just shooting, nothing too strenuous.

For someone who shoots as much as Bedard does, you’d think he would gladly accept being called a shooter. He doesn’t.

“I say it a lot: I don’t feel like I’m a shooter,” he said. “I’m just trying to make the right play. If the shot’s there, I’ll take it. Obviously, the goalies
are good, you got to hit your spot.”
</item 48>

<item 49>
Patrick Kane destinations: Sabres, Red Wings among potential landing spots for free agent winger (Sporting News, 2023-11-27T22:09:43+00:00)
The power play is clicking at 16.4%, 23rd in the NHL. The dynamic winger has collected about 32% of his career points on the man advantage, as he can be an
excellent set-up man or one-time option on the dot.

The problem is that the Sabres have not been as good as expected to start the 2023-24 season. Buffalo was viewed as a team that would take the final step out of
the rebuild this year, given how close the team was to the postseason last year. If Kane has his eyes set on joining a Stanley Cup contender, the Sabres are no
longer atop that list, sitting under .500 and sixth in the Atlantic.

Florida Panthers

The Panthers are the defending Eastern Conference champions, and have continued the momentum from their Cinderella Cup run into the 2023-24 season, sitting tied
for third in the conference and second in the Atlantic.
</item 49>

<item 50>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
Injuries and goaltending have certainly played a factor in their poor play — captain Jared Spurgeon missed the first 13 games of the season, and Filip
Gustavsson’s and Marc-André Fleury’s combined .878 SV% is near the bottom of the league.

They’ve likely been unlucky, too. The Wild have controlled a slim majority of scoring chances and a strong majority of high-danger chances during 5-on-5 play, a
common theme for an organization that’s usually one of the more defensively stout in the league. But the culture around the team is evidently turning sour
quickly, Kirill Kaprizov is on pace for a career-low 26 goals and has a team-worst -10 rating, and most of their depth defensemen have been significant
liabilities. This roster is not all that different from the one that’s finished top-three in the Central Division for three straight seasons, and it’s certainly
one that still has playoff aspirations.

Minnesota has not won since Nov. 7 against the Islanders and is winless in seven straight.
</item 50>

<item 51>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The Wild will certainly get a boost in the standings once Fleury and Gustavsson improve their play between the pipes, a likely scenario given their stellar
performance last season. However, they again have the misfortune of being in quite a competitive division. The Avalanche and Stars occupy first and second place
as expected powerhouses, while the Blues, Predators and Coyotes all sit at or above the .500 mark and, especially in Nashville’s case, have posted decent
underlying numbers.

Hynes’ main job is simply to restore confidence in his group. This roster has the ability to play better — especially Kaprizov, who, despite the okay point
production, has looked like a complete non-factor on some nights. The team also hopes a significant change behind the bench can boost their league-worst penalty
kill, which is operating at just 66.7%. Penalty killing was an area of strength for Nashville last season, who finished tied for fourth in the NHL with an 82.6%
success rate.
</item 51>

<item 52>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.
</item 52>

<item 53>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Tuesday night's NHL national games feature an Original Six matchup at Madison Square Garden in addition to a battle between the team that left and the team that
took its place. The Rangers and Red Wings will face off at the world's most famous arena as both squads have jumped out to hot starts in the new season. Dylan
Larkin is a game-time decision for the Motor City while injuries to the Blueshirts' Adam Fox, Filip Chytil and Igor Shesterkin are making sure some of this
matchup's biggest stars will not be watching in uniform. Regardless, there is still plenty of star power on Broadway in what should feature plenty of offense.

In the late game, the Devils return to Colorado -- the place they left in order to move to New Jersey -- for a showdown with the Avalanche. Again, injuries are
a big part of the storyline here, thanks to Jack Hughes and Nico Hischier out until further notice.
</item 53>

<item 54>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
There isn’t a Josh Allen on the Giants or Vikings.

Minnesota in particular finds itself in an interesting spot, with Kirk Cousins approaching free agency and playing well enough, most likely, to keep the Vikings
from climbing too high in the draft order.

Advertisement

“They already gave a contract to (tight end) T.J. Hockenson, and they are going to give one to (receiver) Justin Jefferson,” an exec said. “Now, what? Are you
going into the rookie pool and taking the fourth-best quarterback in the draft? You can look like Atlanta with (Desmond) Ridder pretty quick.”

6. Two-minute drill: There were already many ways to win a game before the Dolphins invented another one.

The Dolphins were one of 223 teams since 2000 to be favored by at least 13.5 points in a regular-season or playoff game. Seventeen of those 223 teams fell
behind by 14 or more. Six of the 17 won anyway.
</item 54>

<item 55>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury. Hughes
went down after slamming into the boards against the Blues, forcing New Jersey to change things on the fly. The changes have directly impacted Bratt, who was
featured on the top line alongside Hughes. With fellow center Hischier also out with an injury, leaving the Devils fairly weak down the middle and trusting
Michael McLeod to hold down the fort in between Bratt and Tyler Toffoli. As a result, Bratt has been held without an assist in each of the last two games for a
team that's missing a key contributor on offense. And it'll need him to look to score rather than play the role of playmaker. Until the Devils figure things
out, we can't trust this team to do anything positive in the offensive zone.

Win BIG with SuperDraft!
</item 55>

<item 56>
Patrick Kane destinations: Sabres, Red Wings among potential landing spots for free agent winger (Sporting News, 2023-11-27T22:09:43+00:00)
Kane was DeBrincat's main set-up man when the current Red Wings star was potting 40 goals a season. Kane admitted in the past that when DeBrincat was traded
from Chicago to Ottawa it was "disappointing" and "crushing," and this would give him another shot at playing with the talented sniper.

"I’m really curious about the Detroit one," Friedman said. "They have the cap room, they have the space on the roster, they’re doing very nicely so far this
season. As we look on Sunday night, they’re sitting there very much in contention in the Atlantic Division and the Eastern Conference.

"So, I definitely think there’s a lot of people wondering about Detroit and whether or not he ends up there.

The Red Wings have the fifth-most cap space in the league, so money isn't an issue.
</item 56>

<item 57>
Want to help your kids achieve their potential? This expert has some advice (The Age, 2023-12-01T18:00:00+00:00)
Loading Yet in Grant’s book, he illustrates the experiences of those whose potential was not immediately obvious – including his own. Though Grant eventually
became a diver on the junior Olympic team, when he first tried out for his school diving team, it seemed like a hopeless pursuit. The coach told him that the
sport required grace and flexibility, but that Grant walked like Frankenstein, couldn’t touch his toes, and the coach was quite sure his grandmother could
outjump him.

Similarly, though he has become a New York Times bestselling author three times, he failed the mandatory writing test in his first year at Harvard and was
instructed to do a remedial writing course – one reserved for “jocks and international students who spoke English as a fifth language”. “A lot of people believe
passion is out in the world waiting to be discovered,” Grant tells me. “If you can just turn over the right stone you’ll find the thing you love.”
</item 57>

<item 58>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He only fully focused on a potential AFL career last year after previously playing volleyball. Starred with the Calder Cannons and Victoria Metro this year
kicking 33 goals in the talent pathway. His 2km time of 6:05 at the national Draft Combine was a strong indicator of his elite endurance but also his
competitiveness for a player of his size. Is eligible to join the Western Bulldogs under the father-son rule as the son of former Bulldogs utility Matthew
Croft, who played 186 games for the club.

Weapon: Aerial ability

AFL player comparison: Max King (St Kilda)

Jordan Croft has nominated as a father-son prospect for the Western Bulldogs ????
</item 58>

<item 59>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.
</item 59>

<item 60>
Patrick Kane destinations: Sabres, Red Wings among potential landing spots for free agent winger (Sporting News, 2023-11-27T22:09:43+00:00)
It was assumed one of Buffalo, Ottawa or Detroit would take the last step out of the rebuild and become contenders, and through the early going, it looks like
the Wings are that team. Adding Kane could be just the move the Winged Wheel needs to solidify their spot as a playoff contender.

Dallas Stars

One of the dark horse teams that could potentially land Kane is Dallas. The Stars look like one of the best teams in the Western Conference, and there have been
reports that Kane would love to end up with the organization.

The thing is, the Stars already have tons of depth at forward. You could argue that, on paper, Dallas may have the deepest top-nine forward group in the league.
Adding Kane would mean someone like Mason Marchment or Evgenii Dadonov gets dropped to the fourth line, and neither are fourth-line players.

Which is why it makes sense that the rumblings around the Stars and Kane are that the player is much more interested in the team than the other way around.
</item 60>

<item 61>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
What do the Giants’ and Vikings’ ownerships think?

“The best first year can be that 7-10 season where you show some promise,” a veteran coach with experience on a half-dozen teams said. “The Giants won a bunch
of games early, people were heaping praise on the head coach, people get the expectation of what the quarterback can be. But oftentimes, you know different
internally because you know the house you are living in. You are inside the walls every single day and you know where the roof leaks, where the foundation is
cracked.”

Buffalo managed such a situation well. The Bills surprisingly reached the playoffs in their first season under Sean McDermott, a euphoric moment after a 17-year
playoff drought. They went 6-10 the next year, then began a steady rise behind Josh Allen and a top defense. There isn’t a Josh Allen on the Giants or Vikings.
</item 61>

<item 62>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He capped off a great year playing in the Sandringham Dragons premiership team in the Coates Talent League.

Weapon: Height and intercept marking

AFL player comparison: Jake Lever (Melbourne)

Connor O’Sullivan

State: NSW/ACT

State League Club: Murray Bushrangers

Community Club: Albury

Date of Birth: 19/05/05

Height: 198cm

Advertisement

Athletic tall defender who was outstanding for the title-winning Allies at the National Championships winning their MVP award and All Australian honours. He
averaged 17.8 disposals at 86 percent efficiency as well as averaging 8.2 marks and providing great rebound to lead the Allies defence. He has an exciting
natural leap, good speed and elite endurance highlighted with a top 10 placing in the 2km time trial at the national AFL Draft Combine with a time of 6:11.
</item 62>

<item 63>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also named captain of the All Australian team after leading the Allies to their first title. Caught the eye with a strong performance for the Australia
under-18 team in its representative match against Port Adelaide’s SANFL side and backed it up in the second match against Carlton’s VFL side where he collected
24 disposals to be one of Australia’s best in victory. Finished the year as a back-to-back Coates Talent League premiership player with the Sandringham Dragons
where he averaged 31 disposals across nine matches.
</item 63>

<item 64>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was impossible to contain during this year’s National Championships booting 14 majors in four matches for Victoria Metro to top the goal kicking. He also
averaged 12.2 disposals and was rewarded with All Australian selection. He further expanded his game by spending more in the midfield and across half back for
the Eastern Ranges and averaged 20 disposals while kicking 11 goals in nine matches in the Coates Talent League.

Weapon: Speed, agility and leap

AFL player comparison: Shai Bolton (Richmond)

Advertisement

This is just what Nick Watson does ????
</item 64>

<item 65>
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
</item 65>

<item 66>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Daniel Sprong, RW, Red Wings @ Rangers — OVER 2.5 shots on goal

Despite not being featured in the top six, Sprong is holding down the fort on the third line, averaging five shot attempts per game through the first 12. He's
hit the over in five straight, with 18 SOG compared to just eight combined for his two linemates. Sprong does see some work on the second power play unit, but
is clearly willing to fire a lot of pucks at the net with his shoot-first mentality on the third line. With the Rangers still figuring out how to deal with the
loss of Fox, there should be some opportunities there for Sprong to find at least three shots tonight.

3. Jesper Bratt, LW, Devils @ Avalanche — UNDER 0.5 assists

Bratt comes into Tuesday night's action with seven goals and 11 assists in 11 games.
</item 66>

<item 67>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Most of Chicago's shot attempts have been from the outside, making it easy on Tristan Jarry.

8:19 p.m. — It was an uneventful first shift for Bedard, as the Penguins controlled most of the play in the Blackhawks' zone.

8:17 p.m. — Bedard and Crosby square up for the opening face off, and Bedard's debut is underway!

Sidney Crosby 🆚 Connor Bedard on the draw to start this one off.



What a moment. pic.twitter.com/4BPboh8jdp — Sportsnet (@Sportsnet) October 11, 2023

Pregame

7:58 p.m. — Here is how the two sides are lining up for the season opener.

Almost time for hockey!
</item 67>

<item 68>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His ferocious attack on the ball and his opponent helped set the tone for the Allies to dominate the midfield battle throughout the National Championships and
he was rewarded with All Australian honours. Averaged 29.2 disposals, 11.2 contested possessions and a team-high 6.8 tackles across four matches for the Allies
and showed excellent repeat speed and courage as well as clean hands playing both on the inside and outside. His 6:06 effort in the 2km time trial at the
national Draft Combine showcased his elite endurance. He has been part of the Sydney Swans’ Academy since the under-10s.
</item 68>

<item 69>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Penguins.

The game can be streamed through ESPN+, or with Fubo, which offers a free trial.

MORE: NHL predictions 2023-24: Awards, playoff projections, Stanley Cup pick

In Canada, Sportsnet will be bringing the action to TVs. TVA Sports also will have the game for the French-speaking audience. Cord cutters can catch the action
through Sportsnet+.

NHL Opening Night schedule 2023

The Blackhawks-Penguins game is sandwiched between two other opening night matchups.

The first game on the docket was a matchup between the Predators and the Lightning. Tampa Bay is starting this year without star goaltender Andrei Vasilevskiy,
who is expected to miss the first two months of the season after undergoing back surgery.

After the Blackhawks-Penguins contest, the Golden Knights vs. Kraken will wrap up the evening. Before the game in Vegas, the Golden Knights will hold a ceremony
to raise their 2023 Stanley Cup banner.

Tuesday, Oct. 10
</item 69>

<item 70>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 70>

<item 71>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His natural leap was on display at the national Draft Combine where he featured in the top 10 performers in both the standing vertical jump (75cm) and running
vertical jump (91cm).

Weapon: Aerial ability and left-foot kicking

AFL player comparison: Hayden Young (Fremantle)

Angus Hastie

State: Victoria Country

State League Club: Geelong Falcons

Community Club: Geelong West Giants

Date of Birth: 19/09/05

Height: 190cm

An athletic medium defender who sets the play up from defence with his run and ball use. Has shown a preparedness to back himself and take the game on to create
opportunities for his team. His intercepting ability is also a strength which saw him represent Victoria Country in all four matches at the National
Championships. Played 13 matches with the Geelong Falcons in the Coates Talent League this year averaging 20 disposals and 4.6 marks.
</item 71>

<item 72>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 72>

<item 73>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Intercept marking

AFL player comparison: Will Day (Hawthorn)

Advertisement

James Leake

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 02/10/05

Height: 187cm

Medium defender and forward who won All Australian honours as a defender for the Allies at the National Championships in the middle of the season then played
some eye-catching matches as a forward for the Tasmania Devils in the Coates Talent League during a breakout season. With mercurial ability in the air and
uncanny ability around goals he looks to have unique versatility which was on display during 11 matches for the Devils where averaged 16.1 disposals and kicked
17 goals, which included a five-goal haul against the Dandenong Stingrays in Round 17.
</item 73>

<item 74>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.
</item 74>

<item 75>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
pic.twitter.com/6TWSCfXUlR — Steven Ellis (@SEllisHockey) October 11, 2023

8:47 p.m. — Noel Acciari and Tyler Johnson collide in the neutral zone, and Johnson is left leaking from the bridge of his nose. Initially Acciari is assessed a
double minor for high-sticking, but after reviewing the play, no penalty was killed. The officials deemed that the stick was not above the shoulders, therefore,
no infraction.

8:46 p.m. — The Blackhawks kill off the Kaiser minor. Pittsburgh managed to get three shots on Mrazek, but could not build on its lead.

8:43 p.m. — Karlsson dances by Cole Guttman at the blue line, walks into the slot and lets one rip, but Mrazek stares it down the whole way, making the glove
stop on the Penguins defenseman.
</item 75>

<item 76>
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
</item 76>

<item 77>
Patrick Kane destinations: Sabres, Red Wings among potential landing spots for free agent winger (Sporting News, 2023-11-27T22:09:43+00:00)
That's exactly how things played out, as the start of the season began with Kane not on an NHL roster, continuing to rehab from the offseason operation.

It's no secret that Kane isn't the offensive dynamo he was in his prime. He's always been a defensive liability, but it's stuck out more in recent years with
his production on the decline. It's hard to say how much the hip injury affected his short time with the Rangers, but Kane was far from impressive in the Big
Apple.

Regardless, the 35-year-old is ready to get back on the ice, and there are no shortages of interested suitors. Despite Kane's camp trying to keep everything
hush-hush, the rumors and reports have been flying ever since the start of the campaign. It was reported that Kane could make a decision before Thanksgiving,
but the holiday came and went without any news of a deal, however it is expected to come soon.
</item 77>

<item 78>
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
</item 78>

<item 79>
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
</item 79>

<item 80>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Tested very well at the state Draft Combine in Victoria with his 6:12 on the 2km time trial and 3.04 seconds for the 20m sprint rounding off a productive
season. Is eligible to be selected under the father-son rule as the son of former Melbourne utility Nathan D Brown, who played 146 games for the Demons.

Weapon: Footy IQ and ball-winning ability

AFL player comparison: Errol Gulden (Sydney Swans)

Cal Twomey believes father-son prospect Kynan Brown has shown enough to be on an AFL list in 2024, but is more likely to be a later selection in the Draft
</item 80>

<item 81>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Bedard was as advertised in his NHL debut. The Chicago rookie led all players on his club in shots on goal, recording five in his first game on 11 shot
attempts. His 21:29 time on ice was second to only Seth Jones, as Chicago head coach Luke Richardson did not hold the 18-year-old back. He did struggle in the
faceoff dot, winning just two of 13 draws, but it was a strong first impression at the NHL level.

For the Penguins, the third period struggles from last season have seemingly carried over to start this year. Last year, Pittsburgh finished tied for the most
losses after taking a lead into the third period. The crew followed that trend on Tuesday night, letting a 2-1 lead to start the final frame slip away.

Bedard and the Blackhawks are back in action on national TV tomorrow, as Chicago heads to Boston for a Wednesday night date with the Bruins.

MORE: Watch Blackhawks vs.
</item 81>

<item 82>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 82>

<item 83>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Win BIG with SuperDraft! Get a free $10 deposit when you use promo code "TSN"!

4. Cale Makar, D, Avalanche vs. Devils — OVER 2.5 shots on goal

Over this number in each of the last two, fans will want to see Makar shoot the puck a little more after going under in five of the previous six before his
recent hot streak. Facing a Devils team that is reeling thanks to injuries should help Makar's cause with Colorado able to control the play in the Mile High
City. He's the team's top defenseman and a key piece to their power play. Barring something unforeseen, Makar will have plenty of opportunities to find three
shots before the 60 minutes on the clock run out. And with 5.9 shot attempts per game, it's only a matter of time before they start finding their way on goal.
</item 83>

<item 84>
Patrick Kane destinations: Sabres, Red Wings among potential landing spots for free agent winger (Sporting News, 2023-11-27T22:09:43+00:00)
The NHL's version of "The Decision" is almost here, as free agent winger Patrick Kane is closing in on signing with a team for the 2023-24.

OK, maybe it's not nearly as big as LeBron James' NBA free agency choice in 2010, but it's close. Kind of. Maybe tens years ago when Kane was in the middle of
his prime, but I digress.

The three-time Stanley Cup champion has been on the market since the turn of the new season. The longtime Blackhawks star was traded last year at the deadline,
as Kane wanted another shot at winning a Cup and was not interested in sticking around for Chicago's rebuild. He went to the Rangers, but New York was knocked
out in the first round of the playoffs leaving Kane to hit free agency for the first time in his career.

It was announced during the summer that the winger would be undergoing surgery on his hip and that he likely would not make a decision until he was ready for a
return.
</item 84>

<item 85>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Obviously, the goalies are good, you got to hit your spot.”

Blackhawks goalie Arvid Söderblom would certainly call Bedard a shooter. Söderblom and Petr Mrazek have faced more shots from Bedard than any NHL goalies this
season. The challenge for them is that Bedard doesn’t just shoot one way. He’s unpredictable.

“When you have that type of shot, quick release, moves his body and shifts angles on the puck, it makes it harder on the goalies,” Söderblom said. “You just saw
the two goals (on Sunday against the Florida Panthers), both were quick releases. He finds that open net. He’s a pure goal scorer. It’s fun to have on the team
and face him every day and see him take steps, too.”

While goalies are trying to figure out Bedard, he’s been putting in the time to do the same with them. For one, he learned to utilize his teammates more.
</item 85>

<item 86>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Who do you want lining up the potential winning kick in a Super Bowl? It’s an interesting question in a season that has seen Moody and Saints rookie Blake Grupe
miss game-winning tries, but the data could be on the 49ers’ side.

FGA Dist Rookies Veterans <33 Yds 97% [31-32] 94% [330-351] 34-44 Yds 75% [21-28] 82% [230-279] 45-50 Yds 78% [7-9] 73% [121-165] 51-55 Yds 100% [8-8] 59%
[33-56] 56+ Yds 0% [0-1] 55% [6-11] Totals 86% [67-78] 84% [720-862]

The table above shows field-goal make rates in the postseason since 2000 for rookie kickers vs. veterans.
</item 86>

<item 87>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
However, the Blackhawks are challenging the play for offside. Marcus Pettersson had to reach for the puck along the blue line and it may have come out of the
zone.

9:27 p.m. — Jarry keeps the shutout going with two real strong saves on Johnson. A shot from the point was tipped by an unmanned Johnson at the top of the
crease. When that didn't go, he got a second whack at the loose puck, but Jarry stuck out the leg to make the stop.

9:23 p.m. — Reilly Smith, who was traded to the Penguins this offseason from the Golden Knights, is denied by the glove of Mrazek off the rush. Smith had a puck
trickle to him on the right side, but he shot it right into the mitt of the Blackhawks goaltender.
</item 87>

<item 88>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
CLICK HERE for a seven-day free trial to watch the AFL on KAYO

KEVIN SHEEHAN’S TOP 40 PROSPECTS: VIA THE AFL

Kynan Brown

State: Victoria Metro

State League Club: Oakleigh Chargers

Community Club: Blackburn

Date of Birth: 13/01/05

Height: 181cm

Left-footed inside midfielder who has quick hands, sharp vision and creates play through the middle of the ground with his creativity. Had a remarkably
consistent year which included starring for Victoria Metro in the National Championships averaging 22.8 disposals at 80 percent efficiency as well as 4.5
clearances and 4.8 tackles to win their best and fairest award. Had similar numbers with the Oakleigh Chargers in the Coates Talent League including kicking
five goals and having 25 disposals in a best on ground performance in the Wildcard Round win against the Northern Knights.
</item 88>

<item 89>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
Just a couple of years later, Evason had coached the Wild to their best season in franchise history. The 2021-22 campaign saw the Wild break the 50-win mark for
the first time since their inception in 2000, nearly sitting atop the Central Division with 113 points. That was in an extremely tough group with the 119-point
Stanley Cup champion Avalanche and the 109-point Blues, whom the Wild bowed out to in the first round of the playoffs. The Wild remain without a playoff series
victory since the 2015 postseason and have not advanced to a Conference Final since the 2002-03 season.

So with the Wild now far out of the playoff picture with a 5-10-4 record, sitting bottom five in the league in both goals for and against, a change was
necessary and expected.
</item 89>

<item 90>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
A strong showing so far from Bedard, who has not shied away from putting the puck on net. He had four shot attempts in the opening frame, with two on net. The
Blackhawks out-shot the Penguins 17-13, but Pittsburgh holds the lead.

8:51 p.m. — An excellent shift by Bedard nearly results in his first goal. Initially, Bedard shows off his elite release by zipping a shot from long-range, but
it's turned aside by Jarry. Later in the shift, the rookie gets two whacks at a loose puck in front of the crease, but he could not beat the Penguins
goaltender.

Bedard shows off the release, Jarry with the save pic.twitter.com/BKLDkQdpwD — Hockey Realm (@hockey_realm) October 11, 2023

Connor Bedard with his first good scoring chance.
</item 90>

<item 91>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
10:17 p.m. — The Blackhawks are going to their third power play of the night as Matt Nieto is called for hooking just 2:28 into the third.

10:14 p.m. — The third period is a go from Pittsburgh with the Penguins holding onto a one-goal lead.

End of the second period: Penguins 2, Blackhawks 1

9:56 p.m. — Both teams get on the board in the middle frame, but it's the Penguins that remain ahead going into the final period. Bedard looks to be improving
every shift and he picks up his first NHL point on the Donato goal.

9:54 p.m. — An awful line change by the Blackhawks allows Malkin to send Smith on a breakaway, but Mrazek comes up with the stop.

Penguins 2, Blackhawks 1

9:47 p.m. — GOAL!
</item 91>

<item 92>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
First game, first point, first win.

Connor Bedard's debut did not disappoint on Tuesday, as the Chicago rookie helped the Blackhawks to a 4-2 win over the Penguins on opening night.

The Blackhawks rallied from a 2-0 deficit, scoring four unanswered goals to pick up the victory in Pittsburgh.

Bryan Rust opened the scoring for the Penguins in the first period, and a Sidney Crosby goal in the second doubled the Pittsburgh lead. However, Bedard helped
set up a Ryan Donato goal in the middle frame to get the Blackhawks on the board, picking up his first NHL point in the process.

In the third period, Cole Guttman potted the equalizer for Chicago early in the frame. With less than five minutes left in regulation, Jason Dickinson fired
home a shot from the slot that gave Chicago its first lead of the night. A Nick Foligno empty net goal sealed the deal for the Blackhawks.

Bedard was as advertised in his NHL debut.
</item 92>

<item 93>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The Rangers head into this matchup with a makeshift lineup thanks to those injuries that is forcing them to adjust. The absences of Fox and Chytil are a big
deal, however, as Panarin loses Fox, who quarterbacks the power play, and Chytil, who centered his line at even strength. Panarin is still all over the ice for
the Blueshirts though and is even seeing 18.8 percent of his shifts start in the offensive zone, a welcome sight after the last two seasons with Gerard Gallant
behind the bench. The Red Wings have allowed the 11th most SOG this season (371) and the fourth most power play opportunities (46). If the Rangers can play with
the man advantage, Panarin should hit this number with ease, but he's enough of a factor at even strength that we like the OVER here anyway.

MORE SUPERDRAFT DFS & FANTASY PROPS: How to play SuperDraft

2.
</item 93>

<item 94>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Erik Karlsson with the moves 👀#LetsGoPens | #NHLFaceoff pic.twitter.com/GCWVIGnAS2 — Hockey Daily 365 l NHL Highlights & News (@HockeyDaily365) October 11, 2023

8:41 p.m. — Pittsburgh heads to its first power play of the game, as Wyatt Kaiser trips up Malkin in the neutral zone. This will be the first time seeing Erik
Karlsson running the power play for the Penguins, taking Letang's place on the top unit with Crosby, Malkin, Guentzel and Rickard Rakell.

8:36 p.m. — Evgeni Malkin makes a beautiful saucer pass to Jake Guentzel cutting the net, but the Penguins winger could not elevate the puck over the shoulder
of Mrazek. At the other end, Taylor Raddysh took a feed from behind the net, but put it right in the midsection of Jarry.

8:34 p.m.
</item 94>

<item 95>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
The right-hander pitched to a 5.87 ERA over 19 big league outings. His walks more than doubled while his strikeouts fell from 22.9% to a modest 19% clip.

He lost his spot in the Toronto rotation and was eventually demoted to the minors. Manoah never pitched in Triple-A and received an injection in his throwing
shoulder at the end of the season.

How will the Jays proceed after that disastrous year? General manager Ross Atkins said last week the team would give him “a strong leg up” on the No. 5 spot in
the rotation. The Jays aren’t broadcasting a desire to trade him, although they wouldn’t be doing themselves any favors if they declared they were ready to move
on.

Dealing Manoah to bring in a lower upside but higher floor pitcher wouldn’t be a surprise with the Jays firmly in win-now mode. The former first-round pick is
controllable for four more seasons and won’t reach arbitration until next winter.

20.
</item 95>

<item 96>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His time of 5:56 seconds for the 2km time trial at the national Draft Combine was sensational for any player much more so for one of his height. Won the Hunter
Harrison Medal for his performances with the Gold Coast Suns Academy. The medal is presented to the best performed player from the non-Victorian Academy teams
who compete in a selection of matches in the Coates Talent League.

Weapon: Running ability and ball use

AFL player comparison: Mark Blicavs (Geelong Cats)

You're going to want to Read more about Ethan! ????



Hear more from Gold Coast Academy ruckman Ethan Read ahead of the #AFLDraft ???
</item 96>

<item 97>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 97>

<item 98>
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
</item 98>

<item 99>
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
</item 99>

<item 100>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— GOAL! Ryan Donato slips home a rebound from the top of the crease to put Chicago on the board, and Bedard has his first NHL point, picking up a secondary
assist on the tally. Alex Vlasic walks in off the wall and puts a low shot on Jarry that is saved, but Donato pulls the rebound around and by Jarry's leg with
4:23 left in the second.

🚨first NHL assist for Connor Bedard. the full sequence starting with his zone entry pic.twitter.com/JqJMhchiTN — Shayna (@hayyyshayyy) October 11, 2023

Ryan Donato puts home the juicy rebound to give Connor Bedard his first NHL point!#Blackhawks | #NHLFaceoff pic.twitter.com/JbjVPvHVLE — Hockey Daily 365 l NHL
Highlights & News (@HockeyDaily365) October 11, 2023

9:46 p.m.
</item 100>
=== END ITEM id=04a5d54c7001e36e ===
