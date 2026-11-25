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

This file has 5 items.

=== ITEM id=6aedada25ec13d22 ===
Gold sentences and facts:
Sentence 0: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 0.0: Google unveiled new hardware.
  Fact 0.1: Google held its annual hardware event this week.
  Fact 0.2: Google is a search and consumer tech giant.
  Fact 0.3: At its annual hardware event, Google showed off what it has been working on.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed the article for not meeting its standards.
  Fact 1.1: The article was shared widely on social media.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The extent to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google "siphons off" news publishers' readers.
  Fact 2.3: The case argues that Google "siphons off" news publishers' ad revenue.
  Fact 2.4: The case argues that Google obtains these through anticompetitive means.

Retrieved items:
<item 1>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We explain to the whole squad why we are going a certain route. It is not about individual selection,” Springboks coach Jacques Nienaber said shortly after
announcing two changes to his starting side and five in total, after going in with a forwards-heavy 7-1 bench split.

Advertisement

“It is for South Africa. As selectors we know there will be pressure on us if we don’t deliver a result but at the back of our minds we know this selection is
for South Africa.

“The players take it like that. It is not about person or ego, not that they have egos, but you can’t think of that. You have to think about South Africa and
the Springboks.”

Seemingly the Springboks are playing for more than themselves. They’re playing for their country.

It’s a powerful force, one that saw them progress through the knockout stages to reach the final after consecutive one-point victories.

Pressure?
</item 1>

<item 2>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“There were a few tears and I think that’s awesome to see, players showing vulnerability, just opening up about how much this tour and this sort of game means.



“There were some (people) saying this is probably one of the most important games of All Blacks history. (The) All Blacks went through a bit of a hard path to
get here, media, friends, family even criticising us and all that, and we stuck together. We weren’t even said to be in the final and now look at us, we’re
here.

Advertisement

“Different players take it differently and some of the older players probably were hit a little bit harder and it was good to see just a bit of emotion and the
want from them. Seeing that gives all the other players an extra two, three per cent to think ‘OK, we are going to the well and we’ve got to give it our all’.”

The All Blacks will experience a changing of the guard next season.
</item 2>

<item 3>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Four years ago, Rassie Erasmus delivered one of the great rugby speeches ahead of the 2019 World Cup final.

“You don’t have the right today to worry about your mistakes,” Erasmus said before the Springboks sealed a stunning 32-12 victory over England to claim their
third William Webb Ellis Cup.

“If you worry about your mistakes today, you’ve got ego problems.

Facebook Twitter Whatsapp Reddit Email Share

“You are not representing yourself today. You are fighting for the things that happen in South Africa.

“You are tackling for the boy who didn’t get the same opportunities that you got eventually.”

Four years later and the “ego” Erasmus spoke about is what is driving the standards and expectations around the Springboks’ bid to go back-to-back.

“We explain to the whole squad why we are going a certain route.
</item 3>

<item 4>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
The All Blacks are 18-point favourites for the opening semi-final.

It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.

“The past sets us up beautifully for both teams,” Foster said.

“We do know each other but we don’t know each other terms of a Rugby World Cup, so at this stage in the tournament it’s new territory for us and as we’ve seen
World Cups are very different.”

“We have learnt to greatly respect Argentina. They have a rich history of overachieving at the World Cup and have done a fantastic job to get here at the same
level we are, so it is going to be a heck of a game.”

Advertisement

Foster said the All Blacks, who suffered a shock semi-final loss to Eddie Jones’ England in 2019, never consider themselves favourites.

“Firstly, you’ve never heard us say we’re favourites,” he said.
</item 4>

<item 5>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“In the game in 2020 there were things you could pinpoint. You can’t say that one game means we can do it as there are a lot of games where we haven’t.

“That moment has to be just a part of it. What we have done since then, all the success and failures [matters in our preparation]. The other team are heavy
favourites, everyone is expecting a New Zealand and South Africa final. We just have to focus on what is required.”

Cheika has maintained that his Los Pumas side was too tight earlier in the tournament but has begun to free themselves up since qualifying for the knockout
stages – and rewarding their fans by doing so.

On inspiring the country?

“It’s not easy,” he said.

“We want to inspire not just by our results but by our ambition, by our ability to get back up and overcome obstacles.
</item 5>

<item 6>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The pain of the days and weeks after that dark July, as well as the bumps and bruises along the way, including as recently as August when they suffered their
biggest defeat to the Boks to lose the inaugural and prestigious Qatar Cup, still runs deep.

“It was quite special to hear some of the things those certain players said from the heart. You know, you think you know someone but once they open up and
really be vulnerable in front of a team, it was quite special to hear,” back-rower Dalton Papali’i said this week.



“It was truly something special when you hear those boys open up. You know those leaders have been through a lot in their careers, the highs and the lows, the
off-field and on-field stuff, and for them to open up, it sort of makes you feel welcomed and you want to play this game for them.
</item 6>

<item 7>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We don’t want to talk too much about post-game stuff, but that was a tough one, probably the toughest I have had as a coach. He took it the way I expected him
to. He’s a champion.”

Papali’i said the All Blacks were intent on sending their leaders out on a high.

“There’s a few leaders in the team, it’s going to be their last game in the jersey, so it’s a big game for us and even the other players that are staying,
knowing that we want to send those boys out on a high,” he said.

“How much they put into the jersey. We are standing on their…what’s that quote? It’s standing on the shoulders of giants.”

The generational change isn’t just occurring on the field but off it, too.

Advertisement

Foster will depart the All Blacks set up after 12 years, the last four as head coach.
</item 7>

<item 8>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Pressure? The South African laughs at the notion of what pressure is compared to the inequality and challenges the Rainbow Nation continues to experience.

Advertisement

The question is: how can New Zealand’s All Blacks withstand and overcome the hitherto unshakable wave of momentum and spirit the Springboks play with under
Erasmus in this weekend’s World Cup final?

It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and
leaders that will call time on their international careers at week’s end.

The backlash and toxicity that followed their historic home series loss to Ireland was a humbling and public reminder of the expectation of being an All Black.

It’s why Saturday’s World Cup final is personal for the men wearing black.

All Blacks social media managers were forced to delete a thousand comments an hour following the humbling series loss in Wellington.
</item 8>

<item 9>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“It’s the life of this business, it’s a tough game. It’s tough when you’re trying to get your performance right. It’s a tough game when people around you see
things differently, but we’ve learned a lot about ourselves.

“We take massive pride in the jersey and making sure we reach the levels we want to. The rest of that is probably a conversation for another day.

“It hasn’t altered our vision of where we want this team to be. I know we’ve surprised a few people but I don’t think we’ve surprised ourselves.”
</item 9>

<item 10>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The All Blacks will experience a changing of the guard next season.

Indeed, the All Blacks’ social media channels described the team selection for the World Cup final as the “Last Dance”.

2015 World Cup winner Dane Coles will call time on his career, while his tight-five teammates Brodie Retallick and Sam Whitelock, who is bidding to become the
first player to win the Webb Ellis Cup for a third time, will head overseas.

Fellow World Cup winners Aaron Smith and Beauden Barrett will also head to Japan.

Coles, who was labelled by Samisoni Taukei’aho as a “Grandad” type figure in the All Blacks, has even been denied the chance of a farewell send-off after
missing selection.

Advertisement

“Yeah, tough,” said coach Ian Foster of the conversation with the 36-year-old around leaving him out.
</item 10>
=== END ITEM id=6aedada25ec13d22 ===

=== ITEM id=2a9d8a2b950a6f41 ===
Gold sentences and facts:
Sentence 0: The first of these is the fact that tech companies continue to see huge pressure on their valuations.
  Fact 0.0: The first of these is that tech companies continue to see huge pressure on their valuations.
Sentence 1: With a 10% surge in the stock this year, Seeking Alpha's Quant Rating system suggests a Hold, expressing concerns about valuation.
  Fact 1.0: The stock surged 10% this year.
  Fact 1.1: Seeking Alpha's Quant Rating system suggests a Hold.
  Fact 1.2: The Quant Rating system expresses concerns about valuation.

Retrieved items:

=== END ITEM id=2a9d8a2b950a6f41 ===

=== ITEM id=99a8c764347cfe4a ===
Gold sentences and facts:
Sentence 0: He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not.
  Fact 0.0: He also believes that after everything on social media becomes perfectly curated for us, our brains will start to crave things that are not.
  Fact 0.1: He believes that social media becomes perfectly curated for us.
Sentence 1: What we have done since then, all the success and failures [matters in our preparation].
  Fact 1.0: What we have done since then, including all the success and failures, matters in our preparation.

Retrieved items:
<item 1>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 1>

<item 2>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 3>

<item 4>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 4>

<item 5>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.
</item 5>

<item 6>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They can win with offense or defense. Dan Campbell is coach of the year.

4. San Francisco 49ers 6-3 (8)

The 49ers had a surprise offensive get-well game in Jacksonville with Brock Purdy leading the way and all his key supporting weapons doing damage, even without
Christian McCaffrey scoring. The biggest development was the defense going back to dominating a good offense.

5. Baltimore Ravens 7-3 (2)

The Ravens' offense is just fine with the running and passing of Lamar Jackson, but he still can't afford to make big mistakes, which he did vs. the Browns. The
defense needs to pick up the pieces, too, stat, ahead of facing the Bengals on a short week.

6. Miami Dolphins 6-3 (6)

The Dolphins got a much-needed bye to figure out how to beat the better teams in the league.
</item 6>

<item 7>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He also ducked under an end-zone target instead of leaping for it. It's worth noting that a 36-yard crosser he ran against the Chargers flies in the face of
this as he separated from man coverage with speed and technique, then turned upfield and ran outside before racing past a defender and diving for the pylon.

OSBORN: Has a 50% catch rate on the year and that includes whether he faces zone coverage or man coverage. He sees more targets per route run against man
coverage and averages more yards per catch and yards after catch against man coverage. Over his career, targets do seem to come more frequently against man
coverage and he has been more explosive with them.

BEARS: Play zone coverage at the fourth-highest rate of any defense this season (79.2%). And because Cousins is much more explosive of a passer against man
coverage, expect Chicago to keep up their zone defense.
</item 7>

<item 8>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:25 p.m. — The rookie Ivan Pace Jr. finally gets to Fields and drops him for a five-yard loss. It'll be 3rd & long for the Bears.

8:24 p.m. — Fields throws on the run to find Roschon Johnson for a first down, and he quickly follows up with a 13-yard pass to Khalil Herbert. The Bears are
driving, and Fields doesn't have an incompletion yet.

8:21 p.m. — The Bears have taken it into Vikings territory with a string of short gains. For Chicago, that's progress.

8:19 p.m. — Fields finds Moore for a short gain and the first 1st down of the night.

8:16 p.m. — The Bears will start with the ball on the road, looking to build early momentum.

8:00 p.m. — Will the Josh Dobbs show continue tonight?
</item 10>
=== END ITEM id=99a8c764347cfe4a ===

=== ITEM id=e57b76b334a48743 ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 1.0: Every sportsbook jumps on board when a line moves for a legitimate reason.
  Fact 1.1: Every sportsbook jumps on board when odds change for a legitimate reason.
  Fact 1.2: An injury or a roster move is an example of a legitimate reason.

Retrieved items:

=== END ITEM id=e57b76b334a48743 ===

=== ITEM id=89e08552c8ec1386 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The deals are meant to ensure Google is the default search engine on practically every platform.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: The Google deals will probably come up later in the trial.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy on day one.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
housing market set off the dominoes that became the Great Financial Crisis that defined a whole decade economically—and in some ways, our current predicament.

In 2010, the last time existing home sales were so low, the global economy was on life support and the Fed slashed interest rates down to zero to revive the
flatlining patient—fueling an “everything bubble” that inflated all kinds of assets along the way, not the least, housing. The zero-rate regime persisted until
the highest inflation in 40 years surprised many economists and prompted the massive series of rate hikes that have brought us to the once-unthinkable 8%
mortgage.

Some onlookers saw what was coming, for instance Zillow, which warned in May that housing would enter a “deep freeze” if the debt limit standoff failed to
resolve and America defaulted, as that would send mortgage rates up to 8%. Of course, the default was averted, but here we have arrived at 8% mortgage rates
anyway.
</item 1>

<item 2>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Park and her team noted that there are no “noticeable signs” of excess housing development today like there were back then, and households aren’t nearly as
burdened by mortgage debt. Household mortgage debt represented roughly 65% of U.S. consumers’ disposable income in the second quarter, compared to 100% before
the GFC. And the ratio of Americans’ mortgage debt to their real estate assets—also called loan-to-value—was just 27% in the second quarter, compared to over
40% in 2008 and roughly 50% in 2010, Bank of America’s data shows.

New legislation was also enacted since the GFC to help prevent worst-case scenarios in the housing market. One of the most obvious effects of these new laws is
that there are far fewer risky adjustable-rate mortgages today.
</item 2>

<item 3>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
Indeed, existing-home sales prices topped $306,000, a 5% increase since the year began, according to the Case-Shiller U.S. National Home Price Index.

“As has been the case throughout this year, limited inventory and low housing affordability continue to hamper home sales,” NAR chief economist Lawrence Yun
said in a statement. “The Federal Reserve simply cannot keep raising interest rates in light of softening inflation and weakening job gains.” The Fed has raised
rates several times this year, with Chair Jerome Powell arming his “higher for longer” stance.

Declining housing inventory levels also contribute to the drop in housing transactions. Housing inventory is down 8.1% from this time last year, according to
NAR. Some real-estate experts and economists argue that housing affordability is even worse now than during 2008, when a damaging downturn in the U.S.
</item 3>

<item 4>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Housing market pessimists have been sounding the alarm for years about a pending crash in the U.S. residential real estate market. Even before the Federal
Reserve began hiking interest rates to fight inflation last year, pushing mortgage rates to a 23-year high this month, years of surging home prices left some
experts warning that the housing market was a massive bubble ready to pop.

However, despite a widely bearish outlook in the industry, most real estate veterans have avoided arguing that home prices will drop like they did during the
2008 crash that kicked off the Global Financial Crisis (GFC). And so far, that’s been a wise decision.

It took until June 2022 for national home prices to peak, even as mortgage rates soared and purchase applications plummeted amid the Fed’s first interest rate
hikes. And while prices fell more than 5% from that peak by January of this year, they returned to a record high in July.
</item 4>

<item 5>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Adjustable-rate mortgages can lead to higher default rates when interest rates rise, but they now represent less than 5% of total purchase and refinance loans,
compared with over 35% at the peak of the pre-GFC housing cycle.

“We reiterate that we do not expect another housing crash like 2008,” Park and her team concluded after presenting this evidence.

Is it a repeat of the ‘turbulence’ of the 1980s?

According to Bank of America, today’s housing market looks a lot more like the early 1980s than it does 2008. Back then, just like today, home prices had boomed
for years before Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
</item 5>

<item 6>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
And he’s not hopeful we’ll see home prices drop anytime soon: “Absent any negative shocks to the broader economy that would either boost excess supply of homes
on the market or fuel an uptick in unemployment, we continue to expect home prices to rise at a slow pace.” In fact, he predicts we’ll see home prices rise by
1.8% by the end of the year, with a 3.5% increase by the end of 2024.

Looking at more recent housing inventory data makes the issue even more stark. Between September 2018 and September 2023, the average number of homes on the
market dropped a whopping 60% to fewer than 700,000 active listings, according to Realtor.com.

“Unlike the turn of the millennium, house prices today are rising alongside mortgage rates, primarily due to low inventory,” Sam Khater, Freddie Mac’s chief
economist, said in a statement released Sept. 29.
</item 6>

<item 7>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
The average 30-year fixed mortgage rate, the most common type of mortgage in the U.S., has soared from 3.8% in March 2022 to over 7.5% today as a result. This,
in turn, has slowed mortgage purchase applications and caused home sales to plummet, just as it did in the ’80s; but home prices, echoing the dynamic of that
era, have yet to collapse.

One of the key reasons for the resilience of the housing market in both of these periods is demographics. “Noticeably, demographics were favorable back then,
with baby boomers having entered the prime homebuying age,” Park and her team wrote Thursday, arguing millennials are in a similar position today. “Some sales
activity should be supported by millennials reaching the prime homebuying age, and single-family building permits have steadily held up. This can help the
housing market retain some of its momentum without falling apart.”
</item 7>

<item 8>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
29. “These headwinds are causing both buyers and sellers to hold out for better circumstances.”

But with the development that home sales transactions are at their lowest level in the past 13 years coupled with century-high mortgage rates of 8%, many real
estate experts and economists aren’t hopeful that affordability constraints will let up soon.

Out of fear of losing their lower interest rates, current homeowners are resistant to putting their homes on the market, ultimately leading to the abysmal
inventory levels—leaving fewer and fewer properties exchanging hands. Indeed, more than 90% of existing-home owners are locked into mortgage rates below 6%,
Odeta Kushi, deputy chief economist at First American, previously told Fortune.

“These homeowners do not have a financial incentive to sell,” Kushi says. “The combination of reduced affordability and an even stronger rate lock-in effect
suppresses home sales because you can’t buy what’s not for sale, even if you can afford it.”
</item 8>

<item 9>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
And while we haven’t seen home sales activity levels this low since the 2000s, the housing market more closely mimics that of the ’80s, according to a report
published this week by Fortune 500 financial services company First American.

“Today’s housing market isn’t anything like the housing market of the mid-2000s,” First American’s Fleming wrote in a Tuesday report titled “1980s Déjà Vu for
the Housing Market.” “The housing market today is not overbuilt, nor is it driven by loose lending standards, subprime mortgages, or homeowners who are highly
leveraged.”

While some economic factors are stronger today than they were during the GFC, housing affordability is undeniably worse.

“While housing and more generally consumer fundamentals are in a much stronger position today, affordability for the incremental buyer is worse than it was at
the peak in 2006 before the crash,” Roger Ashworth, a managing director at Goldman Sachs, wrote in a credit strategy research paper released last week.
</item 9>

<item 10>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
This can help the housing market retain some of its momentum without falling apart.”

Bank of America economists believe that low inventory of existing homes for sale combined with solid home sales to millennials could help put a “floor beneath
home prices,” but that doesn’t mean there won’t be some near-term pain due to higher mortgage rates.

“With rates likely staying higher for longer, we are cautious of potential turbulence ahead,” the group warned. Bank of America has previously forecast 0% home
price growth and falling home sales for the full year 2023, but it didn’t offer a new prediction in its latest note.

Eventually, as inflation fades leading the Fed to cut interest rates, housing affordability will improve. “At that stage, we should see a more stable and
healthy housing market,” Park and her team wrote. “Until then, hang tight, it may be a bumpy ride.”

This article has been updated to correct a mistaken term used to refer to Jeseo Park.
</item 10>
=== END ITEM id=89e08552c8ec1386 ===
