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

=== ITEM id=6a7672e8159276e2 ===
Gold sentences and facts:
Sentence 0: “It was the millionaires of the Premier League against the youngsters of the Championship,” reflected the former Denmark forward.
  Fact 0.0: A former Denmark forward reflected on a match.
  Fact 0.1: The former Denmark forward said it was the millionaires of the Premier League against the youngsters of the Championship.
Sentence 1: MLB Network's Jon Morosi reports the superstar is "likely to decide on a team within the next week," according to a source.
  Fact 1.0: MLB Network's Jon Morosi reports that the superstar is likely to decide on a team within the next week.
  Fact 1.1: Jon Morosi's report is according to a source.

Retrieved items:
<item 1>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
It will be interesting to see how the most successful team approaches this. Will they be able to find genuine replacements on the market? Only time will tell.
His trading is not worth it for the Yankees, which will further reduce the options in the outfield. With rookies and the farm not compensating for the demand,
it would be a big ask for them to fulfill. With Jasson Dominguez Anthony Rizzo , and DJ LeMahieu

Watch the Story: Ranking the New York Yankees Biggest World Series Failures
</item 1>

<item 2>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
That’s gotta be super disappointing for Mariners fans starved for a title, right?

On the other hand, maybe the M’s know all about Ohtani’s desire to keep the rumor mill quiet, and leaking “news” like this would please Ohtani and his camp.

‘Timing could be right for a Cubs-Ohtani union’

Date: Nov. 16

Source: Jesse Rogers, ESPN

The Cubs were one of the seven finalists for Ohtani the first time around, but lack of the DH in the NL at the time made it hard for Ohtani to choose Chicago.
His primary goal was showing he could both hit and pitch at a high level for an MLB team, and trying to play the outfield instead of just DHing would have been
too much, even for Ohtani.

Now, though? The NL has the DH, and the Cubs are itching to make a sport-shaking signing. Well, another sport-shaking signing.
</item 2>

<item 3>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Juan Soto to Yankees 'close to being finalized'

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network

A trade between the Yankees and Padres for Soto is reportedly "close to being finalized," with right handers Michael King and Drew Thorpe expected to highlight
the return to San Diego. MLB Network's Jon Morosi reported the package will include at least two more players, neither of which will be Alex Verdugo.

Ohtani decision expected to be made before end of weekend

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network

MLB free agency remains largely on hold as teams wait for Ohtani to sign, and though it had at at point been reported he could sign during the winter meetings,
there appears to be a new timeline. MLB Network's Jon Morosi reported Ohtani is expected to make his pick before the end of the upcoming weekend. The winter
meetings end on Thursday.
</item 3>

<item 4>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Bruno is an [effective] player when we have a lot of the ball. Then he is at his best. And there are games against top teams when you have less of the ball. So
then it is about taking benefit from [any attacking] moments and for the rest you have to work hard for the team. And to get a result because that is the most
important, that the team is winning.”

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply.
</item 4>

<item 5>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov. 17

Source: Daniel Kramer, MLB.com

Here’s the key piece of the article: “Industry sources told MLB.com this week that landing Ohtani doesn’t appear to be within the Mariners’ realistic agenda
this offseason.”

In a way, that’s surprising. The Mariners seem to check a lot of boxes for Ohtani. They’re set up to win for a long time with a bunch of young starting pitchers
and a co-superstar in Julio Rodriguez. Plus, Ohtani would get to be part of the quest for the franchise’s first World Series title, he’d get to wear the same
uniform as his baseball hero, Ichiro, and he’s lived there in the offseason. So it must be about the money, and Seattle’s reluctance to even get in the
conversation. That’s gotta be super disappointing for Mariners fans starved for a title, right?
</item 5>

<item 6>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Passan wrote a piece about 10 teams to watch the rest of the offseason, and he threw this little note in the section about the Red Sox, and how they needed to
sign Yoshinobu Yamamoto.

Boston's bats are good, or good enough at least, to avoid having to spend like mad in a down year for free agent hitters -- unless, of course, it's for Ohtani,
who has an affinity for Boston. If the Red Sox don't hit that jackpot, they're backfilled nicely with a system that includes outfielder Roman Anthony, shortstop
Marcelo Mayer and catcher Kyle Teel, among the best trios of offensive prospects in baseball.

It's cool when the best player in the world is a free agent and has an "affinity" for your town.

'Money's great, but winning's better'

Date: Nov.
</item 6>

<item 7>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
4

Source: Ken Rosenthal, The Athletic

While one report earlier Monday indicated Ohtani isn't expected to sign before the end of the Winter Meetings, the process seems to be moving along.

The Athletic's Ken Rosenthal reports Ohtani is "believed" to have met with the Blue Jays at the team's complex in Florida on Monday, far away from the meetings
in Nashville. The Blue Jays have been pursuing Ohtani and Juan Soto in hopes of striking a deal for one of the two stars.

Braves are reportedly 'engaged' with Ohtani's camp

Date: Dec. 4

Source: Jon Morosi, MLB Network

It's not surprising that any and all teams should be interested in Ohtani's services, but imagine the Braves landing the superstar? According to MLB Network's
Jon Morosi, the Braves are "engaged" in talks with Ohtani's camp.
</item 7>

<item 8>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
3

Source: Susan Slusser, San Francisco Chronicle

The Giants have missed out on big names in the past like Aaron Judge and Carlos Correa, but they are looking to turn that around in 2023 with heavy involvement
and Ohtani, among others. Per San Francisco Chronicle's Susan Slusser, teams believe the Giants "might have an edge" in the pursuit of Japanese ace Yoshinobu
Yamamoto.

Dylan Cease trade market heating up

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

One of the biggest names on the market early is White Sox ace Dylan Cease in trade talks. MLB Network's Jon Morosi reported three teams, the Dodgers, Braves and
Orioles, are among several involved discussing a deal for Cease as the winter meetings begin.

Rays, Mariners talking trades in winter meetings

Date: Sunday, Dec.
</item 8>

<item 9>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"Thanks to all the reporters, journos, baseball twitter weirdos, acquaintances and friends who had my back over the last 48 hours," Mintz wrote. "And also, of
course, the BBWAA. Community feels good, humbled to be a part of this one. Now, let's enjoy some ball."
</item 9>

<item 10>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
After the game, Mintz wrote a well-reported story on how the Braves pulled off the improbable comeback, coming back from down four runs at one point, to win the
game, and deep in the article, wrote the following:

Kevin Pillar's young son, sporting a custom "Money Mike" headband, waddled over to his favorite center fielder for a fist bump. Third-string catcher Chadwick
Tromp strutted by Harris' locker offering a congratulatory "Mike is him!" All while Braves shortstop Orlando Arcia cackled emphatically about Harper's
misfortune, bellowing "ha-ha, atta-boy, Harper!" repeatedly as reporters circled the room.

As Mintz pointed out, he was not the only one to allude to the celebration from Arcia. The Washington Post's Chelsea Janes also wrote about it, saying that one
of Harris' teammates was yelling, "Atta boy Harper."
</item 10>

<item 11>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
It appears that might no longer be the case. USA Today's Bob Nightengale reported one high-ranking executive believes the Cubs' optimism of signing Ohtani "has
now significantly wanted." The report stated that leaves the Dodgers, Blue Jays, Angels and Giants as the remaining finalists.

Ohtani has 'positive feeling' about Blue Jays

Date: Tuesday, Dec. 5

Source: Jon Heyman, The New York Post

The Blue Jays reported met with Ohtani on Monday, but even before he went to talk with the team, Ohtani reportedly had a "positive feeling about Toronto," along
with other cities and teams, including the Dodgers. He has been linked to Toronto and Los Angeles throughout the process.

Blue Jays 'believed' to have met with Shohei Ohtani

Date: Monday, Dec.
</item 11>

<item 12>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
It’s not that we don’t think Arizona can beat L.A. at home — the Diamondbacks split six games against the Dodgers at Chase Field this season — but it’s less
than a seven-hour drive from Dodger Stadium to downtown Phoenix, and we’re concerned that all the quirks of that ballpark could be rendered meaningless if
there’s a bunch of Dodgers blue in the crowd.

Advertisement

There’s a lot to like. The pool. The retractable roof. We love it. We hope you prove us wrong. Show up, go nuts, and leave a comment — click to subscribe! —
telling us we don’t know a rattlesnake from a water moccasin. We deserve it.

But if we had to go on the road for a playoff game, at least in this round, we’d choose your ballpark. Consider it a compliment! Or maybe we just like the dry
heat.

No. 7: Texas Rangers Team Venue Opened Capacity Att.
</item 12>

<item 13>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Dodgers officials declined comment when asked if Ohtani had taken a physical, the final step before a deal can become official.

Significant questions linger about Ohtani’s future. He will not pitch in 2024 as he recovers from a September operation to repair his right elbow’s ulnar
collateral ligament. He underwent Tommy John surgery in 2018. Neither Ohtani, who has not taken questions from reporters since August, nor his agent, Nez
Balelo, nor the Angels have disclosed the exact nature of the second surgery, but the Los Angeles Times has reported it was a second Tommy John procedure.

Balelo has stressed that Ohtani remains committed to both pitching and hitting in the future. “Shohei loves to pitch,” Balelo told reporters in September.
Ohtani will attempt to return to the mound in 2025. His camp has not revealed at what point Ohtani would consider giving up his dual career and focusing on
learning a different position.
</item 13>

<item 14>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec. 8, 2017, six years to the day.

Dave Roberts says Dodgers met with Ohtani

Date: Tuesday, Dec. 5

Source: Alden Gonzalez, ESPN

Dodgers manager Dave Roberts told reporters Tuesday that the team met with Ohtani for 2-3 hours over the weekend. Roberts said Ohtani is "clearly" the Dodgers'
top priority, adding that the team is confident the two-time MVP will pitch again. Roberts wasn't able to offer a timeline for Ohtani's decision.

Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec. 5

Source: Bob Nightengale, USA Today

The Cubs have been viewed as one of Ohtani's likeliest destinations for a while. It appears that might no longer be the case.
</item 14>

<item 15>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
The Athletic's Ken Rosenthal reports Ohtani is "believed" to have met with the Blue Jays at the team's complex in Florida on Monday, far away from the meetings
in Nashville. The Blue Jays have been pursuing Ohtani and Juan Soto in hopes of striking a deal for one of the two stars.

Shohei Ohtani, Yoshinobu Yamamoto decision not expected until after winter meetings

Date: Monday, Dec. 4

Source: Jon Heyman, The New York Post

The baseball markets are waiting for the first major domino to fall this offseason in free agency, but it does not appear as though that will happen before the
winter meetings conclude. Jon Heyman reports that neither Shohei Ohtani not Yoshinobu Yamamoto are expected to get a deal done before the final meetings on
Wednesday.

Wade Miley back with Brewers on 1-year deal

Date: Monday, Dec.
</item 15>

<item 16>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Since he was a teenager, Ohtani has ignored suggestions that he focus on only one pursuit.

Ohtani demonstrated his potential as the American League Rookie of the Year in 2018, but his two-way hopes were delayed after his first elbow surgery. It was
not until 2021 that the full flower of his ability bloomed. He has won the American League MVP in two of the past three seasons; in the intervening season, he
led all American League pitchers in strikeout rate while hitting 34 home runs with an .875 OPS. To create a comparison for him involves inventions that sound
freakish. “It’s like if Judge went out and was a 20-game winner as well,” former teammate Kole Calhoun said.

Ohtani will not pitch in 2024 after undergoing Tommy John surgery, but his agent stressed he’s committed to returning to the mound. (Photo: Michael Owens /
Getty Images)

Ohtani is committed to being a starting pitcher.
</item 16>

<item 17>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
MLB's winter meetings are always a prime time for moves to happen as many of the top executives, players and agents all come together at the same spot.

Last year's winter meetings wound up being explosive. Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.

This year's winter meetings could wind up being similarly busy. There is a growing sense Shohei Ohtani could sign during the winter meetings, and a move of that
magnitude could kick off the next wave of moves. There's also a number of top players on the trade market, including star Padres outfielder Juan Soto.

Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.

WINTER MEETINGS: Shohei Ohtani rumors, Juan Soto trade destination come to forefront

Sporting News will be tracking all the latest news and rumors from the 2023 Winter Meetings.
</item 17>

<item 18>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
Here's what you need to know about the controversy.

Bryce Harper Atta Boy controversy, explained

It all begins with the play. In Game 2, Harper led off the top of the ninth in a one-run game with a walk. J.T. Realmuto flew out to center for the first out,
but on the next play, Nick Castellanos lifted a deep fly into right-center. It appeared it would be extra bases, and if Harper took off, he'd be able to score
the tying run on the play.

The problem for the Phillies was that Harris made a leaping grab up against the wall to rob Castellanos of the hit. Harper had well since rounded second on his
way to third, but the throw from Harris missed both cut-off men by second. Austin Riley, however, ran in and scooped the ball and made an impressive throw to
Matt Olson to nab Harper and end the game.
</item 18>

<item 19>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
(Photo: Michael Owens / Getty Images)

Ohtani is committed to being a starting pitcher. If he cannot stay healthy enough for that role, he could aid his new team as a reliever. He closed the final
game of the World Baseball Classic, securing the crown for Japan by striking out Trout. His four-seam fastball averaged nearly 97 mph in 2023; the velocity of
the pitch figures to improve in short bursts.

Advertisement

Even if Ohtani never pitches again, his value as a hitter is immense. In 1920, his first season exclusively as a hitter, Ruth led the American League in homers,
on-base percentage and slugging percentage. In 2023, while still making 23 starts with a 3.14 ERA, Ohtani led the American League in the same three categories
as Ruth. He swatted 44 homers with a career-best 1.066 OPS. He did this while making starts and dealing with a torn ligament in his elbow.
</item 19>

<item 20>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
20

Source: Ken Rosenthal, The Athletic

Buried at the bottom of Rosenthal’s piece on what an Ohtani contract might look like — incentives, opt-outs and all sorts of stuff — is this little bit of
reporting:

An agreement at the winter meetings in Nashville the first week of December would be the ideal outcome for MLB, but not necessarily for Ohtani. He perhaps will
be better off waiting for the signing of Japanese righty Yoshinobu Yamamoto, who is expected to be posted for free agency as soon as Monday, triggering a 45-day
window for teams to negotiate with him. Losing out on the 25-year-old Yamamoto, who could command more than $200 million, might put certain teams on tilt,
improving the landscape for Ohtani.

Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov.
</item 20>

<item 21>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: Bob Nightengale, USA Today; Ben Nicholson-Smith, Sportsnet

Ohtani had a change of plans, apparently. Hours after MLB Network's Jon Morosi reported that the two-way star was headed to Toronto for a meeting on Friday, USA
Today's Nightengale and Sportsnet's Nicholson-Smith reported that Ohtani is at his home in Southern California.

The Blue Jays are still finalists to land Ohtani but his landing spot is not official yet.

Ohtani headed to Toronto on Friday

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

Ohtani might not have officially picked a team, but he is headed to Toronto, per MLB Network's Jon Morosi. It had been previously reported he had picked the
Blue Jays, though several reports have since denied that he has made a decision.

Report: Ohtani picks Blue Jays over Dodgers

Date: Friday, Dec. 8

Source: J.P.
</item 21>

<item 22>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“I think that it is important sometimes that you have to go through those moments to realise what it takes. There are elements that happen in a football match
and elements that happen in the buildup to a football match, so you have to experience all those things if you want to be successful.”

The win against Everton on Sunday was defined by two teenagers, Alejandro Garnacho and Kobbie Mainoo.
</item 22>

<item 23>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Morosi reports Shohei Ohtani is "likely to decide on a team within the next week," according to a source.

Status quo on Yankees' Juan Soto trade talks

Date: Sunday, Dec. 3

Source: Jon Heyman, New York Post

The Yankees reportedly are pushing hard to acquire Juan Soto from the Padres, but Heyman says the teams haven't spoken since San Diego requested a six-player
package in return for Soto and Trent Grisham. That ask included pitchers Michael King and Drew Thorpe, who the Yankees aren't inclined to move. Stay tuned...

Shohei Ohtani field growing smaller

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

It has been expected since before he hit free agency that Ohtani would set the record for the largest fully guaranteed contract in North American sports
history, with Passan reporting the deal could reach as much as $600 million.
</item 23>

<item 24>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
It will include the shift of Oswaldo Peraza from third baseman to second. However, Peraza could not do well this season, even after all the positives. His
involvement in unique positions could be a risk for the team.

ADVERTISEMENT Article continues below this ad

Moreover, free agents Whit Merrifield and Adam Frazier have not done well. Given that the Yankees finished second last in total hits and had over 1400
strikeouts, losing a player with the best record in both parameters will not be ideal. Notably, Torres has a 14.6% strikeout rate, with only a handful of
players recording a lower rate. He also covered a lot of bases this season to finish second to Judge. All these factors make him a player that the Yankees
should keep as they look for a change in 2024.

ADVERTISEMENT Article continues below this ad

not in the best shape, losing Torres could haunt the Yankees further.
</item 24>

<item 25>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 25>

<item 26>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Who will hand him that contract has been much more of a mystery. Little has been leaked about the top suitors for Ohtani beyond the usual suspects. Passan
reported that at least three teams initially believed to be in the running — the Mets, Rangers and Red Sox — are believed to be out now. The teams most likely
to still be in the mix are the Dodgers, Cubs, Blue Jays and Angels, with the Giants' status in the race for Ohtani unknown.

Star players could be traded

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

The free agency pool outside of Ohtani, Cody Bellinger, Blake Snell and Yoshinobu Yamamoto is a bit lacking in star power, but the trade market could ensure
several big-name players are on the move this offseason.
</item 26>

<item 27>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Certain things [here] are different,” shrugs Ten Hag when asked why the No 1 is taking time to settle in England. “OK it is 11-v-11, and it is about attacking,
defending, transition. But there is more: the approach to the game, the speed of the game, the intensity. The varieties in styles. So there is absolutely a
difference and players need to adjust, need time to do so.”

Last season’s derby at Old Trafford ended in a 2-1 win for United. Bruno Fernandes’s equaliser was controversial with Marcus Rashford offside but not so the
goal was chalked off as he had not touched the ball. So far this term the captain has scored two goals in 12 United appearances and Ten Hag has no doubt he can
be a major factor in Sunday’s showdown.

“If he does [score] again, I’ll be happy,” the manager says. “Bruno is an [effective] player when we have a lot of the ball.
</item 27>

<item 28>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
Also, given he is currently in arbitration with the Yankees, it adds to his trade chances. The reason is that the team won’t want to lose the opportunity to
cash in some money from his trade. The decision appears promising as the team seeks a shift, but the data tells a different story. Gleyber Torres’s Exit Becomes
the Talk of the Town as the Yankees Look for a Rebuild The New York Yankees have been in the news since the league stage ended. First, because of the
possibility of a third-party audit, which was later shunted out. Later, the comments from Aaron Judge brought a lot of attention to the workings of the team,
conflicts, and divisions between the core four. Now it’s the assumptions of changes with the idea of cutting down the $300 million budget for the next season.
One name that is making the rounds for an exit is Gleyber Torres.
</item 28>

<item 29>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: J.P. Hoornstra, Dodgers Nation

The Blue Jays have reportedly won the sweepstakes for Ohtani. According to Dodgers Nation's J.P. Hoornstra, Ohtani has selected the Blue Jays. The official
details have not been reported, though the report indicated it will shatter Mike Trout's record 12-year, $426.5 million deal.

However, Sportsnet's Ben Nicholson-Smith reported Ohtani has not yet made his decision, and there is not a timeline, though he noted Toronto is a finalist.

Decision from Ohtani 'imminent'

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

The wait could soon be over. MLB Network's Jon Morosi reported Ohtani is expected to make his decision soon, reporting it could come as early as Friday.
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec.
</item 29>

<item 30>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Do the Dodgers have the budget to afford Ohtani and Trout on the same roster?

ADVERTISEMENT Article continues below this ad

Shohei Ohtani’s next projected contract value is predicted to be around $500–$600 million. If it goes through, he will break Trout’s biggest MLB deal record of
12 years, $426.5 million. Now, we all know that the Dodgers have deep pockets; they have managed to maintain quite a costly lineup this season valued at $283
million. But can they afford two of baseball’s biggest superstars at full value?

Now, with Trout’s $426.5 million tagline, the slugger has an annual salary of about $35.45 million. He still has seven years left on his Angels contract, so get
him in blue, the Dodgers will have to pay $248.15 million plus any extension value.
</item 30>

<item 31>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
While he won't pitch again until 2025, Ohtani's power and on-base ability gives Los Angeles the most dangerous trio of hitters in the sport.

MORE: How Dodgers' 2024 payroll looks after handing out monster deal to Shohei Ohtani

Red Sox swing deal for Cardinals' Tyler O'Neill

Date: Friday, Dec. 8

Source: Jon Heyman, New York Post

Days after trading Alex Verdugo to the Yankees, the Red Sox have acquired OF Tyler O'Neill from the Cardinals in a deal that sends pitching prospects Nick
Robertson and Victor Santos to St. Louis.

O'Neill enjoyed a breakthrough 2021 season with the Cardinals, batting .286 with 34 home runs and a .912 OPS, but he's struggled to recapture that level of
production over the last two seasons and is set to be a free agent after 2024.
</item 31>

<item 32>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
One of those 10,000 Rakes definitely belongs to Royce Lewis. 💪 pic.twitter.com/3gRJp2i8eb — MLB (@MLB) October 3, 2023

Our take: We have only a small sample of postseason games at Target Field, but, boy, Twins fans can make some noise. After Game 1 against Toronto, Twins manager
Rocco Baldelli said, “I thought the place was going to split open and melt, honestly. It was out of this universe out there on the field. The fans took over the
game. They helped us win today.” The next day, they did it again.

Advertisement

Early in Game 2, Correa told Sonny Gray that, because of the crowd noise, Jays baserunners couldn’t hear their third-base coach screaming, “BACK!” So, with
Guerrero on second and a full count to Bo Bichette, they executed a perfect timing pick.
</item 32>

<item 33>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Shohei Ohtani’s singular pursuit of history, one man’s quest to rewrite the baseball world’s understanding of what is possible, reached another summit on
Saturday when he agreed to the largest contract in the annals of major North American team sports, a 10-year, $700 million contract with the Los Angeles
Dodgers, his agency CAA announced.

Advertisement

Ohtani announced his decision on Instagram. The deal ends years of feverish speculation about Ohtani’s future. Ohtani, a 29-year-old two-way sensation, has
captivated the industry since he left Japan for Major League Baseball heading into the 2018 season. He has done things that appeared impossible in the modern
era, feats that harkened back to Babe Ruth. As he traveled the country with the Los Angeles Angels this past summer, fans serenaded him with recruiting pitches.
When he entered free agency, a dozen teams lined up, curious to see if they could meet his eye.
</item 33>

<item 34>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 34>

<item 35>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
With that, one has to wait and watch who serves as the best fit.

ADVERTISEMENT Article continues below this ad

With the facts broken down, it is a matter of time that the teams narrow down their best options. MLB world will welcome 2024 with high expectations. With new
management, the fans expect teams to come back with great vigor and ferocity.

Watch This Story: Every MLB Team That Has Never Won a World Series
</item 35>

<item 36>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
The “intensity” of baseball in L.A. isn’t the same as it is in, say, St. Louis or Boston or any other city where athletes are the biggest celebrities in town.

Advertisement

But by the middle innings, Dodger Stadium is packed, and by the late innings of playoff games, the place has the over-the-top intensity of a Michael Bay car
chase (Mookie Betts as Bumblebee). More than one opposing pitcher this season had to tweak his PitchCom device to deal with the late-inning noise that came from
such a massive ballpark, one that once again held the largest total attendance in baseball this year.

“You’ve got a four-deck stadium,” Freddie Freeman said this summer. “It’s the only one in baseball. You’ve got 50,000 people, every single night here. For them
to go out of their way to make you feel good when you’re doing your job, it means a lot. I do appreciate it.
</item 36>

<item 37>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
“The crowd was incredible,” Gray said after the game. “They were incredible yesterday. They were incredible today from the moment I stepped on the mound an hour
before the game to the moment — they’re probably still out there.”

So, Twins fans have swarmed their team with support. But is it intimidating? (It’s certainly better than its predecessor, the Metrodome, but it doesn’t have the
same potential for overwhelming volume.) For now, Target Field strikes us as more of an awesome place to play than one that strikes fear in the opponent. But
this month Twins fans could prove us very wrong.

No. 5: Houston Astros Team Venue Opened Capacity Att. (2023) Astros Minute Maid Park 2000 41,168 37,683

Memorable postseason moment: Albert Pujols off Brad Lidge? Jorge Soler to the moon?
</item 37>

<item 38>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Imagine a lineup consisting of Ohtani, Ronald Acuna Jr., Ozzie Albies, Austin Riley and Matt Olson?

Sources: Braves are among the teams still engaged with Shohei Ohtani’s camp, as I reported in this @MLBNetwork segment. @MLB pic.twitter.com/lP4lISzttN — Jon
Morosi (@jonmorosi) December 4, 2023

Ohtani has already received several $500+ million offers

Date: Dec. 3

Source: Jon Heyman, NY Post

It's no surprise the bidding for Ohtani is heating up, with the star expected to sign soon. And New York Post's Jon Heyman reported Ohtani has already received
several deals that have topped $500 million, which could mean he winds up being the first $600 million player in MLB history. He added there are believed to be
five teams in the running for Ohtani.
</item 38>

<item 39>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.
</item 39>

<item 40>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
Those in Philly would like to forget the Astros’ combined no-hitter in Game 4 of last season’s World Series, so instead we’ll concentrate on Roy Halladay’s 2010
no-hitter in Game 1 of the NLDS against the Reds. In his first postseason start, Halladay allowed only one baserunner, when Jay Bruce walked with two outs in
the fifth inning.

Our take: Look, Philadelphia fans have thrown snowballs at Santa Claus and batteries at J.D. Drew. Their Phanatic mascot is iconic, but part of his charm is the
way he – it? – taunts and antagonizes opponents. It’s a real love-it or hate-it place to play. The Phillies and their fans are capable of producing
unforgettable moments, like Wednesday night when Bryson Stott hit a grand slam and the ballpark provided the only commentary necessary.
</item 40>

<item 41>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."
</item 41>

<item 42>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Roberts said Ohtani is "clearly" the Dodgers' top priority, adding that the team is confident the two-time MVP will pitch again. Roberts wasn't able to offer a
timeline for Ohtani's decision.

Pair of pitchers could be signed after Ohtani deal

Date: Tuesday, Dec. 5

Source: Jon Morosi, MLB Network

MLB is largely in a holding pattern as it awaits news on Ohtani. MLB Network's Jon Morosi reported the Rays could wind up trading Tyler Glasnow shortly after
Ohtani signs, listing the Cubs and Reds as top contenders for the ace. He also reported Yoshinobu Yamamoto could sign following Ohtani's deal, and that the Red
Sox have him among their top priorities.

Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec.
</item 42>

<item 43>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Erik ten Hag smiles when asked if André Onana can be the “holding midfielder” for Manchester United described in Uefa’s technical report of the goalkeeper’s
display for Internazionale in May’s Champions League final.

“First the job of a keeper is to save, is to [guard] his goal, that’s his main job and don’t forget this,” says United’s manager. “But, yeah, he is a keeper who
we can use as an extra player and if we are developing [more] he can be more progressive and even higher up the pitch – an extra player, so that’s an
advantage.”

Onana’s last game for the Italian club was the 1-0 defeat by Manchester City in Istanbul as United’s crosstown rivals claimed a first Champions League and so
clinched the treble. Yet if he has recent experience of facing Pep Guardiola’s side, the 27-year-old will be a derby debutant against City in Sunday’s 191st
edition of the local squabble that will attract a global audience of millions.
</item 43>

<item 44>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Those two would be the biggest supporting bats in the lineup around Ohtani. However, both are free agents after the 2025 season, and given the added payroll of
signing Ohtani, it's possible Toronto would view keeping the other two phenoms as too pricey.
</item 44>

<item 45>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Yes, Ohtani does have magnificent hitting form as well. However, the team already has hitters like Freeman and Betts, who have scored 211 and 179 hits this
regular season, respectively. Speaking of hitters, why will they still want Trout to join the team despite having better-than-average hitters on the team?

Why will the Dodgers sign an injury-prone Trout?

Mike Trout might have been absent for most of the 2023 season, but his career records boast the story of a legendary hitter who has given more than his salt’s
worth to his current team. He has bagged 1624 hits and 940 RBIs with an 85.2 WAR and .301 BA across his 12-year-long baseball career. Unfortunately, Trout has
become a victim of an injury streak, starting in 2022.
</item 45>

<item 46>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 46>

<item 47>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec. 5

Source: Bob Nightengale, USA Today

The Cubs have been viewed as one of Ohtani's likeliest destinations for a while. It appears that might no longer be the case. USA Today's Bob Nightengale
reported one high-ranking executive believes the Cubs' optimism of signing Ohtani "has now significantly wanted." The report stated that leaves the Dodgers,
Blue Jays, Angels and Giants as the remaining finalists.

Ohtani has 'positive feeling' about Blue Jays

Date: Tuesday, Dec. 5

Source: Jon Heyman, The New York Post

The Blue Jays reported met with Ohtani on Monday, but even before he went to talk with the team, Ohtani reportedly had a "positive feeling about Toronto," along
with other cities and teams, including the Dodgers. He has been linked to Toronto and Los Angeles throughout the process.
</item 47>

<item 48>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
The Los Angeles Angels have gone through a tough year. After a heartbreaking 2023 season, they now have to cope with the impending exit of their star two-way
star, Shohei Ohtani. The team is also open to trading their nine-time Silver Slugger hitter Mike Trout, should he request it. In a huge miss, a team that
couldn’t do well even with two of the league’s biggest talents on their roster may lose said stars at the same time. Will they choose to stay in LA?

America’s Favorite Video Today

The Halos’ cross-town rivals, the Los Angeles Dodgers, have been eyeing the dynamic Troutani duo for a long time. However, despite fans’ expectations, the
Dodgers’ Ohtani-Trout conquest might not take place—and for good reasons.

Do the Dodgers have the budget to afford Ohtani and Trout on the same roster?
</item 48>

<item 49>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Uefa’s description of Onana intrigues because it suggests that what Guardiola introduced to the domestic game – a No 1 in Ederson who performs as a de facto
extra defender to maintain fluidity along the backline – has evolved further. The goalkeeper was so advanced as to be a No 6 (rather than, the technical report
posited, a “sweeper-keeper”) who can take up residence in the middle third of his half as his team pushes deep into enemy territory.

Onana did so against City for Internazionale but has this encounter with the champions come too soon for him to function as “the extra man” Ten Hag identifies?
“It also depends on the opposition, how strong is the press, for instance, and how good is the team around him,” the United manager says.

“So if it’s possible – in my philosophy he has to do it every game [but] it is not always possible, so you have to adapt and this what I make clear.
</item 49>

<item 50>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"That's taking away from true reporters and true journalists. Where were you on the random Tuesday in April in Cincinnati when this team was playing in
Cincinnati? Where were you on a rain delay, and you have to sit in the press box for all of this time? It's ridiculous."

"And then some jackoff comes in at the end of the season that gets a credential, God only knows why. And the clubhouse is a sacred space." Alanna Rizzo went off
on Jake Mintz's clubhouse reporting on MLBN's "High Heat" Thursday. pic.twitter.com/6dooqeLPnh — Awful Announcing (@awfulannouncing) October 12, 2023

Rizzo's comments immediately drew heat from the baseball journalism world, including from Janes, who posted a thread defending Mintz. Janes reported that
cameras and recorders were recording, and that Arcia yelled the comments while they were recording and with more than a dozen reporters around.
</item 50>

<item 51>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Instead, the Royals will hope the veteran Smith, who had solid underlying numbers in 2023, can serve as a reliable late-inning arm after both Scott Barlow and
Aroldis Chapman were traded before the deadline this past season.

Shohei Ohtani agrees to mega contract with Dodgers

Date: Saturday, Dec. 9

Source: Shohei Ohtani, Jeff Passan

Two-time MVP Shohei Ohtani announced on Instagram that he is signing with the Los Angeles Dodgers, spurning the Blue Jays and remaining in the city where he
became a sensation with the Angels.

ESPN's Jeff Passan reports Ohtani has agreed to a colossal 10-year, $700 million contract shattering the MLB record for both total value and average annual
value.

Ohtani joins a Dodgers lineup that already features two former MVPs in Mookie Betts and Freddie Freeman, both of whom were NL MVP finalists in 2023.
</item 51>

<item 52>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
When the Phillies beat the Braves in Game 4 to eliminate Atlanta and advance to the NLCS, players in the Philadelphia clubhouse celebrated by repeating, "Atta
boy Harper."

While the comments became a spark for the Phillies, it also became a subject of controversy. Players, including the Blue Jays' Kevin Gausman, decried the
situation as a mistake by Mintz to report the comments, which were said out loud in the locker room rather than said in an interview setting. Others defended
the situation, saying that anything in the clubhouse is open to be reported on.

MORE: Why Arcia's comments motivated Harper after NLDS Game 2

But perhaps the biggest backlash came from MLB Network's Alanna Rizzo, who blasted Mintz on "High Heat" for reporting on the incident. Following a BBWAA
statement, Rizzo both publicly and privately apologized to Mintz for her reaction.

Here's what you need to know about the controversy.
</item 52>

<item 53>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
It is clear that the Guardians are not keen on getting someone from within MLB; they are open to outside hires as well. As long as the new candidate can follow
Terry’s principles.

One applicant slated for an interview was Craig Albernaz. He is the catching and bullpen coach for the San Francisco Giants, according to Paul Hoynes of
Cleveland.com. Despite lacking Major League managerial experience, the 40-year-old has led two championship-winning Tampa Bay Rays teams. Another likely
candidate could be Carlos Mendoza, who is the bench coach for the New York Yankees. Though he too lacks managerial experience in MLB, the 43-year-old is
familiar with the Yankees system, suiting well for Cleveland.

Speaking Of NY, The New York Mets Are On The Lookout

In his first season as manager, Buck Showalter guided the Mets to a 101-61 record. With that, he earned him the 2022 NL Manager of the Year Award.
</item 53>

<item 54>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
He still has a lot left to give to the diamond for approximately 6 more years. If the Dodgers have the chance to draft them, should they go with Ohtani, Trout,
both, or neither? Let us know your opinions in the comment box.

Watch This Story: Shohei Ohtani’s Two-Way Dream: Before Injury, He Always Wanted Both!
</item 54>

<item 55>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
While the offseason has gotten rolling, trade season hasn’t quite arrived. The Tigers acquired Mark Canha from Milwaukee in the biggest swap of the first couple
weeks. Otherwise, the GM Meetings simply served to set the stage for future transactions.

That’ll surely involve some notable trades. Unlike the summer deadline, projecting which players will change hands is difficult at this stage of the offseason.

Aside from the A’s, there aren’t any teams clearly in rebuild mode. Clubs like the Rockies, Royals, White Sox, Nationals and Angels have uphill paths to
contention, yet none has indicated they’re prepared to punt on the 2024 season entirely.

The Padres, Twins, Rays, Brewers and Guardians are all aiming to contend but have questions about their payroll. That’s standard operating procedure for Tampa
Bay, Milwaukee and Cleveland. San Diego and, to a lesser extent, Minnesota have shown a willingness to spend in recent years but are scaling back — in both
cases, likely tied to their uncertain local television rights contracts.
</item 55>

<item 56>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
The Hot Stove dominoes have started to fall, but everyone who loves baseball — fans, players, coaches, front-office types and everyone else — is waiting for
free agent Shohei Ohtani’s decision. Where will the global superstar sign his next contract?

He’s played his entire six-year MLB career with the Angels, but his teams haven’t even sniffed the postseason and Ohtani’s made his desire to win crystal clear,
so he won’t be back. Will he chase every dollar possible, going solely by the highest offer? That seems unlikely, too. Remember, Ohtani wants to win.

Lots of teams can meet Ohtani in that middle ground, offering both lots of money — he will absolutely sign the largest contract in baseball history — and lots
of chances to compete for a World Series title (or two or three). It’s a long list.

MORE: Ranking 15 possible destinations for Shoehei Ohtani

So, the baseball world waits for news.
</item 56>

<item 57>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Unfortunately, Trout has become a victim of an injury streak, starting in 2022. He got his hands, left groin, back, and foot injured five times in the entire
season; in 2023, he faced two hand injuries in two consecutive months.

Despite his injury-prone history, the more logical option for the Dodgers would be to choose Trout over Ohtani; the team will not be able to afford them both at
the same time. Trout comes with a cheaper price tag of $35.45 million compared to Ohtani’s projected $48.63 million. The Dodgers could use his All-Star form in
between injuries if he still keeps getting them in 2024. With Shohei Ohtani, the Dodgers would be wasting a year letting him play at-bat, with a high chance of
getting him further injured and missing more pitching time.
</item 57>

<item 58>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Shohei Ohtani’s predicted $535 million for 11 years (according to a Sports Illustrated segment) will bring the Dodgers down to pay him a good $48.63 million
annually.

The Dodgers already have six players whom they have to pay above $10 million a year: Freddie Freeman ($27 million), Mookie Betts ($25 million), Clayton Kershaw
($20 million), Chris Taylor ($15 million), Max Muncy ($13.5 million), and J.D. Martinez ($10 million). Among the top-paid players, Clayton Kershaw is exiting
the team this season. That, along with a few other player offloads, will free up enough space in the Dodgers payroll to support two $25 million or three $10
million p/a players on their team. Neither Trout nor Ohtani falls within that price range, so keeping both together at the same time would not be a feasible
option for the Dodgers right now.

Pitchers or Hitters?
</item 58>

<item 59>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
The Max Fried-Jorge Soler game in 2021 happened in Houston. The iconic moment at Truist Park could be Eddie Rosario’s dazzling catch at the wall in Game 4 of
the 2021 World Series, but the most memorable — even though it didn’t lead to a championship — might be rookie Ronald Acuna Jr. hitting a grand slam off Walker
Buhler in Game 3 of the 2018 NLDS, which happened to be the first postseason game ever played at the ballpark.

Our take: Obviously, there are issues bigger than baseball to deal with here. The war chant and Tomahawk Chop are offensive and problematic, but we’re not going
to resolve that issue in the next four weeks. In fact, it’s all about to come front and center — again — as the best team in baseball opens its postseason with
a roster that would be plenty intimidating in a Little League park with metal bleachers.
</item 59>

<item 60>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
after newsletter promotion

And clarity is perhaps what this team has been lacking most of late: clarity of selection, clarity of formation, clarity of mission and motivation. How much
prestige should we really attach to the Nations League, anyway? Is it primarily an Olympic qualification shootout or something worth winning in its own right?
England ended up trying to answer these questions on the hoof, and somehow it was only when their options had finally clarified down to one – survival or
oblivion – that they recovered their poise and verve.

This entire post-pandemic international cycle – a Euros, a World Cup, an Olympics and another Euros squeezed into three years – feels like a kind of sadism
dressed up as a panorama of opportunity. There is a school of thought out there that a fallow summer might not have been the worst thing for this team: time to
reflect and process, time to breathe, time to refresh.

Instead, they now travel to Glasgow on Tuesday for one last push. The minds are still frazzled.
</item 60>

<item 61>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Isn’t it better to have a consistently wounded soldier than risk losing an overperforming one in a game of chances?

ADVERTISEMENT Article continues below this ad

Why would Mike Trout switch to another Los Angeles team?

Trout has been pretty persistent about sticking with the Los Angeles Angels from the very start. The team has been transparent about not stopping him should he
choose to play for another team, but he does not want to do that. However, it seems that the Angels will have a hard time affording a player like Trout in the
foreseeable future, as they have to win a good number of games to stay in the playoffs and keep a steady income of revenue throughout the seasons. Considering
their 73-game win record this season, that seems like a pipe dream right now.

ADVERTISEMENT Article continues below this ad

Trout is 32 years old, which means that he has already hit the MLB average retirement age of 30.
</item 61>

<item 62>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
As the seconds ticked down at the end of a Champions League season‑ender that felt like a gentle, even quite tender, act of sporting euthanasia, the only noise
inside Old Trafford was the sound of the Bayern Munich fans singing an impressively sustained version of Football’s Coming Home. A little later they sang Is
This A Library? Actually, no, it’s more of a museum. They chanted “Auf Wiedersehen” at the departing members of the home support who, frankly, had done pretty
well to stick it out to that point.

The good news for Erik ten Hag at the end of this low-fi 1-0 defeat is that United’s players didn’t lie down and die at Old Trafford. Instead they stood up and
died, running hard and creating a kind of simulacrum of a functioning elite team.

This was not a group of players in revolt or drained of spirit. They looked like what they are, a muddled creation, stitched together out of parts and off-cuts.
</item 62>

<item 63>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

“After the break, we were more compact and certainly more energetic, hungrier and more determined,” Milan’s coach, Stefano Pioli, told DAZN. “That is a great
sign when playing every three days and it was a fully deserved victory.”

Milan suffered a potential injury blow as the midfielder Ruben Loftus-Cheek limped off in the first half. “It would be a shame if we lost Ruben, he felt a
twinge near the groin area,” Pioli said. “Unfortunately, injuries tend to come when players are in good shape.”
</item 63>

<item 64>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
That said, the skipper market is abuzz!

Starting With The Los Angeles Angels

ADVERTISEMENT Article continues below this ad

After Joe Maddon was relieved of his managerial duties, Phil Nevin signed up as the interim manager for the remaining 106 games of 2022. Post that, he went on
to sign a one-year contract to manage the Angels in 2023, and the interim tag was lifted. Nevin’s 2024 club option was rejected by the Halos, who underwent
partial two seasons of 119-149 (.444) play under his supervision. Since this was a non-negotiable, the seat stands vacant.

The biggest challenge in this scenario is that the Angels star, Shohei Ohtani is yet to reveal which team he would be joining. In case the Angels re-sign the
two-way phenom, the manager would get to head the roster with the 2021 MVP and Mike Trout. A once in a lifetime opportunity.
</item 64>

<item 65>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 65>

<item 66>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
“We’ll see what the appeal brings. The mentality is key for me, we took a knock but there have been a lot of knocks here over the last couple of years.

Advertisement

“But the team mentality is growing. I can’t emphasise it enough, the commitment to each other and the connection is a powerful thing because I know we’ve got
quality.”

Eddie Howe’s side are proving one of the most inconsistent in the league, albeit with a lengthy injury list.

In the last month, they have defeated Manchester United, Arsenal and Chelsea – as well as drawing at French heavyweights PSG in the Champions League – but have
now lost comfortably to two cellar dwellers in Everton and Bournemouth.

“We have to do better,” said Howe.

“There are reasons behind every performance and we have struggled to change players. That is a big miss with the quality of players out. The longer you do it,
the harder it gets.
</item 66>

<item 67>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Even if you’re not playing well, if the team is not playing well, or you’re not playing to the standards you expect, you still have to win – so find a way to
win, be pragmatic.”

United have been so of late, claiming victories over Sheffield United, Brentford and FC Copenhagen in their last three matches. But Ten Hag’s “depends on the
opposition” and “how good is the team around him” comments sound like hints. In contrast to Inter’s Simone Inzaghi, the United manager may harbour doubts about
deploying Onana high up the pitch against City.

The Dutchman is still building his team and may be wary of the uneven form of United’s new keeper. Bought for £47.2m, Onana arrived on a ticket of being a
high-grade shot-stopper, technically impressive too with the ball at his feet. But until impressing against Brentford and Copenhagen, Onana had bombed, costing
United goals and points.
</item 67>

<item 68>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Here's the latest on Ohtani's free agency pursuits.

MORE: Projecting Shohei Ohtani's contract in free agency

Shohei Ohtani news: Latest updates, rumors for MLB free agency

Ohtani agrees to 10-year, $700 million deal with Dodgers

Date: Saturday, Dec. 9

Shohei Ohtani announced Saturday that he has agreed to a contract with the Dodgers. It's a colossal 10-year, $700 million deal with "significant deferrals," so
while Los Angeles won't pay him $70 million per year, the franchise will be paying him beyond those 10 seasons.

While rumors connected Ohtani to the Blue Jays on Friday, it's the Dodgers who hand him a record-shattering contract and pair the two-time MVP with two former
MVPs in Mookie Betts and Freddie Freeman.

Ohtani not headed to Toronto, at his home in Southern California

Date: Friday, Dec.
</item 68>

<item 69>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
By Chad Jennings, C. Trent Rosecrans and Stephen J. Nesbitt

In one American League Wild Card Series, the whole thing turned on a play designed around crowd noise. It was too loud at Target Field, Twins shortstop Carlos
Correa realized, for the third-base coach to warn Blue Jays baserunners about a developing pickoff play. The Twins used the noise to their advantage, picked off
Vladimir Guerrero Jr. and secured a two-game sweep.

Advertisement

In the other AL Wild Card Series, empty seats were everywhere. The Rays are an excellent team that thrives on their underdog status — low payroll, injured
players, they always find a way — but playing in front of two of the smallest postseason crowds in the past 100 years, the Rays were swept by a Rangers team
that had nine fewer wins in the regular season. The indifference was deafening.

Home-field advantage, it turns out, just isn’t the same from one ballpark to the next.
</item 69>

<item 70>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
The baseball season for the New York teams didn’t turn out to be their most successful one. Both the New York Mets and the New York Yankees failed to make it to
the playoffs and came under scrutiny as the highest spenders in the league. But the level of accounting has come higher for the 27-time winners because of the
highest expectations. The management has comprehended the situation and is actively seeking to make changes for the 2024 season. Be it trimming the squad and
releasing a few players, including superstar Gleyber Torres, they look to bring about changes. But given that they are already short on options, it brings up
the question: Is the move justified?

America’s Favorite Video Today

Although it was a decent season for the second baseman, the lack of consistency brought Torres under scrutiny for changes for the upcoming season. Also, given
he is currently in arbitration with the Yankees, it adds to his trade chances.
</item 70>

<item 71>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Only one team could secure Ohtani’s services. He will now be compensated for both his immense talent and his unparalleled star power. His contract eclipsed the
$360 million record for free agents set last winter by New York Yankees slugger Aaron Judge and also surpassed the record-setting $426.5 million extension from
Ohtani’s former Angels teammate Mike Trout. His achievement exceeded even those outside of baseball, topping the $450 million contract inked by Kansas City
Chiefs quarterback Patrick Mahomes. Ohtani has outpaced even soccer star Lionel Messi’s $674 million contract — signed in 2017 when he was with FC Barcelona.

His individual brilliance was not enough to lift the Angels into the postseason. With the Dodgers, Ohtani will now have an opportunity to add collective
hardware to his trophy case. The Dodgers have won the National League West in 10 of the past 11 seasons, topped 100 victories in five of the past six full
seasons and won the World Series in 2020.
</item 71>

<item 72>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
We have been promised very little of that, though. ESPN’s Jeff Passan reported this: “If visits between Ohtani and a team are reported publicly, it will be held
against the team.”

So, yeah. Probably won’t hear a ton, but that won’t stop the speculation. And it won’t stop us from passing along what precious few nuggets that are scattered
about.

MORE: Way-too-early MLB Power Rankings for 2024

Shohei Ohtani free agent rumors and updates

Ohtani has an 'affinity' for Boston, it seems

Date: Nov. 27

Source: Jeff Passan, ESPN

Hope everyone enjoyed the Thanksgiving holiday. Us baseball writers are pleased that Ohtani didn't decide to sign while the turkey was still warm. This isn't
exactly "breaking news" because rumors have been swirling, but it's an aside from an in-the-know reporter that just might give one fan base a little too much
hope and excitement.
</item 72>

<item 73>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
And all the effort is showing.

According to Joe Pompliano, “The NFL will bring in $20 billion in revenue this year, more than the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1
combined.” Coming to the strategy part, a curious mind would perhaps think can they take a leaf out of the NBA’s playbook and reinvigorate its appeal as it aims
to conquer international audiences?

Can the NFL take a leaf out of the NBA’s grisly past and glorious recovery?

The NFL is already the world’s richest league. Calling it an incredible feat would be a gross understatement. Consider this: the English Premier League, which
is broadcast in over 212 territories, has a revenue of just $6 billion, while football games are primarily viewed in America only. However, the NFL does not
just want to stop at $18 billion.
</item 73>

<item 74>
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
</item 74>

<item 75>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The hardcore football fans are pretty much done with the NFL harping about Travis Kelce and Taylor Swift. Reality TV star Rachel Lindsay, a longtime NFL fan,
recently slammed the “obnoxious” coverage, highlighting the “exhausting” uptick in Swifties obsessing over football. But that’s just one example of how fans are
finding this fixation a little too disrespectful towards the sport.

America’s Favorite Video Today

From the league’s humble beginnings in 1920 to the $18 billion juggernaut that it is today, the ride has been rocky with more than just a few small bumps along
the road. But with the ambitious target that NFL Commissioner, Roger Goodell, has set for 2027, is there a new strategy in the 104-year-old league’s master plan
to dominate the world sporting scene?

Why the ‘Traylor’ saga will be the key to Roger Goodell’s $27 billion target?
</item 75>

<item 76>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
The Yankees are seeking a shift in their strategy, which has exceeded their initial expectations. This shift encompasses alterations in both offensive and
defensive personnel, and surprisingly, it involves Gleyber Torres. Although it is a reboot button from Brian Cashman for a better 2024, should the Yankees
refraining from including Torres? The answer is simple: they should indeed because of his worth as a player. It may sound huge as they expect Torres to earn $15
million, but it is not a big ask for the multibillion-dollar Yankees. The 2009 champions can easily fulfill this request without constraining their spending.

ADVERTISEMENT Article continues below this ad

Plus, Torres is the sole option for the Yankees at second base. With limited options available in the market, the team is in the circumstances to spend more.
And if the Yankees are for other infielders, they may have to reshuffle to make their place in the squad.
</item 76>

<item 77>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
A once in a lifetime opportunity. On the flip side, if they lose Ohtani, the roster would need a solid revamp. Not to mention the Angles are infamous for their
below-par farm system and playoff drought spanning 9 seasons.

With that out in the open, what are the possible options for the Halos? Ray Montgomery could be an option, considering that he began his tenure with the Angels.
Walt Weiss could potentially be a decent fit, too. Weiss has been a part of a successful team in the Atlanta Braves recently, having been their bench coach
since before the 2018 season. Additionally, among other options are Ron Washington and Clayton McCullough. Though these options are open, no solid information
has been revealed yet as to who would fill up the position.

Moving On To The Houston Astros

After his fourth season with the Astros and his 26th big league managerial season, Dusty Baker announced his retirement last week.
</item 77>

<item 78>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Dodgers sign Yamamoto to $325 million contract

Date: Tuesday, Dec. 19

Source: ESPN's Jeff Passan, YES Network's Jack Curry

The Yamamoto sweepstakes are over, as the Dodgers have reportedly signed the Japanese ace to a 12-year, $325 million. It's the largest MLB deal for a pitcher,
surpassing Gerrit Cole's deal with the Yankees by $1 million. Los Angeles' incredible offseason continues, as Yamamoto joins Shohei Ohtani on the defending NL
West champs.

Giants out of Yamamoto sweepstakes

Date: Thursday, Dec. 21

Source: Alex Pavlovic, NBC Sports Bay Area

The Giants made a hard push to land Yamamoto, but according to NBC Sports Bay Area's Alex Pavlovic, San Francisco has been notified that the team is out of the
bidding to land the Japanese ace.
</item 78>

<item 79>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
The Giants are considered to be an unknown.

In addition to the teams, Passan also reported Ohtani is expecting a contract somewhere between $550 and $600 million, which would be the largest guaranteed
contract in North American sports.

Blue Jays could alter roster with Ohtani signing

Date: Nov. 30

Source: Jon Heyman, New York Post

The Blue Jays are one of the teams believed to still be among the strongest in the mix for Ohtani. The New York Post's Jon Heyman reported there could be some
changes if Ohtani signs with the Blue Jays, however, reporting that some rivals believe Bo Bichette and Vladimir Guerrero Jr. could hit the trade block if
Toronto lands the star.

Trading the two infielders might seem counterproductive to signing Ohtani in the first place. Those two would be the biggest supporting bats in the lineup
around Ohtani.
</item 79>

<item 80>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Well, another sport-shaking signing. They already poached Craig Counsell from the rival Brewers to be their new manager, and it would be shocking if they didn’t
make a corresponding player-acquisition move. Ohtani’s at the top of the list.

Ohtani could potentially sign before Winter Meetings

Date: Nov. 11

Source: Alden Gonzalez, ESPN

Pretty much everyone in baseball would LOVE for this to happen. The team that winds up signing him, of course, will kick the Ohtani hype machine into gear ASAP.
Not only is there merchandise to sell, but the front office will use that signing to help lure other potential free agents, maybe even for a “play with Ohtani”
discount. But even for the other teams, knowing that Ohtani is off the board relatively soon into the offseason would allow them to shift to other priorities,
like trading for Juan Soto or deciding which free-agent pitchers to pursue with the money that might have been earmarked for Ohtani.
</item 80>

<item 81>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
4

Source: Jon Morosi, MLB Network

It's not surprising that any and all teams should be interested in Ohtani's services, but imagine the Braves landing the superstar? According to MLB Network's
Jon Morosi, the Braves are "engaged" in talks with Ohtani's camp. Imagine a lineup consisting of Ohtani, Ronald Acuna Jr., Ozzie Albies, Austin Riley and Matt
Olson?

Sources: Braves are among the teams still engaged with Shohei Ohtani’s camp, as I reported in this @MLBNetwork segment. @MLB pic.twitter.com/lP4lISzttN — Jon
Morosi (@jonmorosi) December 4, 2023

Korean star Jung Hoo Lee to be posted Monday

Date: Monday, Dec. 4

Source: Jeeho Yoo, Yonhap News

Star Korean outfielder Jung Hoo Lee is set for a move to MLB.
</item 81>

<item 82>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
after newsletter promotion

This is what Ten Hag has overseen in recent weeks as the protracted sale of the club has entered a new phase, with Sir Jim Ratcliffe’s bid to buy 25% for an
inflated £1.3bn in return for control of football policy expected to go through imminently. But despite any implications this may have for the futures of Ten
Hag’s players, the manager himself does not think the matter is a distraction.

Bruno Fernandes volleys home for Manchester United against Burnley – one of two goals he has scored for the club this season. Photograph: Matthew
Peters/Manchester United/Getty Images

“Of course, they are committed to the club but especially they are committed to this team,” Ten Hag says. “They want to be successful and are therefore aware
you need [to focus on] the team, your teammates. That is how the players are thinking and they are not thinking about strategic reviews or structures or
whatever.”

On Sunday at 3.30pm we will find out just how focused United’s players are.
</item 82>

<item 83>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
He did this while making starts and dealing with a torn ligament in his elbow.

Only one stage remains for Ohtani. He has never played a postseason game in the majors. As the Angels foundered in recent years, Ohtani became more vocal about
his desire to play for a winner. Now he has the chance to make a different kind of history.

Required reading

(Photo: Stacy Revere / Getty Images)
</item 83>

<item 84>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 84>

<item 85>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
'Money's great, but winning's better'

Date: Nov. 21

Source: Max Scherzer, Rangers pitcher on Foul Territory

OK, so this isn't necessarily a rumor, but an example of how much players would love for Ohtani to join their team. Scherzer was a guest on Foul Territory on
Monday.

"Money's great, but winning's better."



Max Scherzer shares his pitch for Shohei Ohtani to join the @Rangers 🏆



▶️ https://t.co/6KoSRqSgk2 pic.twitter.com/PHW2jHWj6C — Foul Territory (@FoulTerritoryTV) November 20, 2023

Maybe Ohtani won’t sign quite so soon after all

Date: Nov.
</item 85>

<item 86>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 86>

<item 87>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Shohei Ohtani and Juan Soto have already found new homes, but much of the MLB offseason has yet to play out.

Japanese pitcher Yoshinobu Yamamoto is the focus of many teams as he prepares to sign a massive deal, while Blake Snell, Cody Bellinger and Jordan Montgomery
remain on the market. The trade market is also expected to be active once more dominoes fall, with Dylan Cease, Corbin Burnes and Tyler Glasnow among starters
who could be moved this winter.

The Winter Meetings are over, but because of the timeline of major decisions by Ohtani and Yamamoto, there could be a flurry of activity leading up to the end
of the year.

MORE: Why Shohei Ohtani signed with Dodgers over Blue Jays, other suitors

Sporting News is tracking the rumors, signings and trades as MLB free agency continues. Follow along below for the latest news.
</item 87>

<item 88>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Ohtani has never played a postseason game in his big-league career.

“My sense is that he wants to be the best ever,” said St. Louis Cardinals outfielder Lars Nootbaar, who played with Ohtani last spring in the World Baseball
Classic, “but I don’t think he would ever publicly say that.”

Advertisement

Deferrals effectively lower the average annual value of Ohtani’s contract, giving the Dodgers greater flexibility as they attempt to navigate Major League
Baseball’s luxury-tax system, which includes penalties for teams that exceed certain thresholds.

A team’s luxury tax-payroll is calculated according to players’ average annual values, and discounted by deferrals. Mookie Betts, for example, has $115 million
deferred in his 12-year, $365 million contract with the Dodgers. Under the league’s calculations, the deferrals lower his average annual value from $30.4
million to $25.5 million.
</item 88>

<item 89>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
A young Orioles roster might be at risk of being shell-shocked on the road, but at home, this team is going to be treated as heroes from the start.

"Go crazy, Baltimore. You are the champions of the American League East." Kevin Brown on the call as the Orioles clinch the American League East and the No. 1
seed in the American League. pic.twitter.com/Q2eXPe4uYL — Awful Announcing (@awfulannouncing) September 29, 2023

Let’s just hope, when they go on the road, manager Brandon Hyde doesn’t forget to use his best reliever in a must-win game. That could be pretty bad.

No. 2: Los Angeles Dodgers Team Venue Opened Capacity Att.
</item 89>

<item 90>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
This was a win that felt a lot tighter than it really was: United inexplicably hanging on at the end despite enjoying 28 shots and an xG tally of 4.07. For long
periods United played some of their most authentically thrilling football of the season, a doctrine of relentless attack that had Diogo Dalot flying forward
from right-back even as the minutes ticked down.

It was also a game that felt more chaotic than it really was. In fact the slippery, slingy nature of the play was largely by design: United hollowing out their
midfield in order to throw bodies forward, Chelsea hollowing out theirs in order to leave men up for the breakaway. It was a game that hung on United’s
determination to keep allowing Chelsea counterattacks, and Chelsea’s determination to keep messing them up. It was a game, above all, that hung on the curious
main character energy of Scott McTominay.
</item 90>

<item 91>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Yates, who will be 37 when the season begins, is headed to the Rangers on a one-year deal, per Rosenthal. The right-hander, an All-Star closer for the Padres in
2019, posted a 3.28 ERA in 61 appearances for the Braves in 2023.

Devenski, 33, has agreed to a new one-year deal with the Rays that includes a club option for 2025, according to Passan. Devenski began 2023 with the Angels but
was released in late August and signed with Tampa Bay. He posted a 5.08 ERA in 29 games for the Angels and dropped three full runs off that number in nine games
with the Rays.

Dave Roberts says Dodgers met with Ohtani

Date: Tuesday, Dec. 5

Source: Alden Gonzalez, ESPN

Dodgers manager Dave Roberts told reporters Tuesday that the team met with Ohtani for 2-3 hours over the weekend.
</item 91>

<item 92>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
All eyes since the start of the offseason have been on Shohei Ohtani. And for the most part, there hasn't been much to see.

The two-way phenom is the most sought-after free agent in MLB history, with his combination of talent and marketability making him a franchise-changing player.

And the bidding for Ohtani could be winding down soon. Multiple reports have indicated Ohtani could sign at some point during the winter meetings, which are
scheduled to take place from Dec. 3 through Dec. 6.

Many of the top teams are expected to be in the mix for Ohtani. The Dodgers and Cubs are among the top teams expected to be aiming to sign the two-time AL MVP,
and plenty of others should continue to compete through the end to land him.

Here's the latest on Ohtani's free agency pursuits.
</item 92>

<item 93>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
We remember Onana also diving out of the way of the ball in Istanbul. Casemiro’s red card, lost leads in Manchester, Marcus Rashford unfairly convicted of a
venomous stamp in Copenhagen, when frankly, he’s just not that kind of semi-interested footballer.

And yet with both group games blank at half-time United were still just one goal away here from going through to the last 16, which really would have made for
an excellent punchline. It wasn’t to be. For all their own flaws Bayern are basically a good version of whatever it is United are attempting to piece together.
Kimmich‑Goretzka‑Musiala is an A grade version of Amrabat‑McTominay‑Fernandes. Manuel Neuer is the thing Onana is trying so earnestly to be. Bayern actually did
go ahead and sign Harry Kane.

Harry Kane (centre) and his Bayern Munich teammates applaud fans after their Champions League victory at Old Trafford.
</item 93>

<item 94>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
He added there are believed to be five teams in the running for Ohtani.

Ohtani decision coming this week

Date: Dec. 3

Source: Jon Morosi, MLB Network

With the winter meetings set to begin in Nashville, it seems a resolution is near for Ohtani. MLB Network's Jon Morosi reports the superstar is "likely to
decide on a team within the next week," according to a source.

Ohtani race starting to clear up

Date: Dec. 1

Source: Jeff Passan, ESPN

Two days before the start of the winter meetings, the picture on Ohtani is beginning to clear a bit more. ESPN's Jeff Passan reported the Rangers, Red Sox and
Mets appear to be out of the running for Ohtani, while the Dodgers, Cubs, Blue Jays and Angels remain in the mix. The Giants are considered to be an unknown.
</item 94>

<item 95>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Pavlovic added that it is very likely that Yamamoto ends up either in Los Angeles with the Dodgers, or New York with the Mets or Yankees.

Yamamoto met with Giants, won't decide on team until after Christmas

Date: Tuesday, Dec. 19

Source: Andrew Baggarly, The Athletic and Bob Klapisch, Newark Star-Ledger

San Francisco has been among the teams most heavily believed to be in on Yamamoto, and according to The Athletic's Andrew Baggarly, the team met with him in San
Francisco last week. Baggarly reported the Giants have been willing to "blow teams out of the water" to land Yamamoto, but noted he believes the crowded picture
for the Japanese ace could make it more challenging for the Giants to be the winner of the Yamamoto sweepstakes.

And fans will have to wait a bit longer before hearing a final resolution on his decision.
</item 95>

<item 96>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
11

Source: Fabian Ardaya, The Athletic

According to The Athletic, Ohtani will defer a whopping $680 million of his $700 million contract from the Dodgers, receiving only $2 million per year.

While it's important to note the deal will still count as $46 million annually toward the luxury tax thresholds for Los Angeles, Ohtani is still giving the
organization much more flexibility than expected over the course of the deal.

The two-time MVP will receive $68 million per year, without interest, from 2034-43, making the present-day value of the contract much less than it originally
appeared.

MORE: Complete breakdown of Shohei Ohtani's Dodgers contract, by the minute

Giants reportedly meet with Yamamoto

Date: Monday, Dec. 11

Source: Buster Olney, ESPN

ESPN's Buster Olney reports the Giants met with Japanese ace Yoshinobu Yamamoto on Sunday, Dec. 10.
</item 96>

<item 97>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
2023 has been a topsy-turvy year for many teams. While some have outshone others, these 5 teams are at a crossroads, with a common stifling problem. Amid the
heavy restructuring that many of the MLB teams have seen of late, there are pentagonal vacancies for managers. The New York Mets, the Los Angeles Angels, the
Houston Astros, the Cleveland Guardians, and the San Diego Padres are in dire need of filling up the managerial position as the year ends.

America’s Favorite Video Today

While some team representatives skim through names within the MLB world, some hope to look for an outside hire. It could be a risky state for all of them.
Nevertheless, there are many possibilities and roadblocks one might encounter. Be it budgetary issues, past performance, or availability of candidates, these
teams have not had success in putting a pin on one candidate. That said, the skipper market is abuzz!
</item 97>

<item 98>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
11

Source: Los Angeles Dodgers

The Dodgers have officially announced their signing of Ohtani, calling him a "once-in-a-generation talent and one of the most exciting professional athletes in
the world."

"Dodger fans, thank you for welcoming me to your team. I can say 100 percent that you, the Dodger organization and I share the same goal - to bring World Series
parades to the streets of Los Angeles," Ohtani said in a statement.

The Los Angeles Dodgers have signed two-time American League Most Valuable Player Shohei Ohtani to a 10-year contract. pic.twitter.com/mggYwgp01i — Los Angeles
Dodgers (@Dodgers) December 12, 2023

Ohtani to defer $680 million of $700 million Dodgers deal

Date: Monday, Dec.
</item 98>

<item 99>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Pitchers or Hitters? Who Will the Dodgers Prioritize This Free Agency?

The Dodgers have been in dire need of pitchers for a few seasons. Their current highest strikeout rate by a pitcher in a single postseason is 137 SOs by Clayton
Kershaw, followed by 119 SOs by Bobby Miller and 117 SOs by Julio Urias. With Shohei Ohtani’s high strikeout rate of 167 this season, he’ll be an invaluable
addition to the team. However, there’s a little hiccup that could keep that from happening.

Ohtani will not be available to pitch throughout the 2024 season due to an elbow ligament injury he’s currently recovering from. Given the pitcher’s unavoidable
absence, it will not be advisable for the Dodgers to hire him until 2025. Yes, Ohtani does have magnificent hitting form as well.
</item 99>

<item 100>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
But MLB Network's Jon Morosi said it's unlikely the Jays will land both star players, reporting they might land one of the two, but they will not add both this
offseason.

Ohtani has already received several $500+ million offers

Date: Sunday, Dec. 3

Source: Jon Heyman, NY Post

It's no surprise the bidding for Ohtani is heating up, with the star expected to sign soon. And New York Post's Jon Heyman reported Ohtani has already received
several deals that have topped $500 million, which could mean he winds up being the first $600 million player in MLB history. He added there are believed to be
five teams in the running for Ohtani.

Decision near for Ohtani

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

Nothing gets the hot-stove crowd excited like a big move during the winter meetings, and the biggest move of the offseason could come this week in Nashville or
soon after.
</item 100>
=== END ITEM id=6a7672e8159276e2 ===
