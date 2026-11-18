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

This file has 6 items.

=== ITEM id=6fca2a41bd8a2f0d ===
Gold sentences and facts:
Sentence 0: It was a Michigan quarterback that led the Buccaneers to the promised land last time so perhaps lightning strikes twice.
  Fact 0.0: A Michigan quarterback led the Buccaneers to the promised land last time.
  Fact 0.1: Lightning may strike twice.
Sentence 1: Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh.
  Fact 1.0: The source for this is the Big Ten.
  Fact 1.1: The Big Ten is reviewing lengthy written responses.
  Fact 1.2: The written responses are from Michigan.
  Fact 1.3: The written responses are from Jim Harbaugh.

Retrieved items:
<item 1>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
According to ESPN.com, Stalions allegedly purchased tickets to more than 35 games involving Big Ten and potential CFP opponents.

Photos also surfaced os a person who resembles Stalions on the Central Michigan sideline during the Chippewas' opener against Michigan State on Sept. 1. Central
Michigan has yet to confirm who that person was, another layer to a wild Stalions story that has supplied endless meme and conspiracy material on social media.

Stalions might be the fall guy who saved Michigan from further wrath from the Big Ten and NCAA or the reason the Wolverines' national championship season
unraveled. That fate is yet to be determined, but at minimum he will be one of those controversial names we'll never forget, like Harvey Updyke and Nevin
Shapiro.

Jim Harbaugh

On one hand, Harbaugh could serve his second suspension within one season and further NCAA punishment could be coming down the line.
</item 1>

<item 2>
2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR (CBSSports.com, 2023-11-17T20:07:09+00:00)
The Falcons have added some older veterans at the position but it is not a viable long-term solution.

Round 1 - Pick 10 Laiatu Latu EDGE UCLA • Sr • 6'5" / 265 lbs Projected Team Washington PROSPECT RNK 11th POSITION RNK 1st After moving on from Chase Young and
Montez Sweat, Washington has a need for rushing the passer. Laiatu Latu may end up being the best out of this draft class.

Round 1 - Pick 11 J.J. McCarthy QB Michigan • Jr • 6'3" / 202 lbs Projected Team Tampa Bay PROSPECT RNK 21st POSITION RNK 4th Baker Mayfield is replaced in
Tampa Bay with J.J. McCarthy. It was a Michigan quarterback that led the Buccaneers to the promised land last time so perhaps lightning strikes twice.
</item 2>

<item 3>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 8, 2023

Tuesday, Nov. 7

Michigan says three other Big Ten schools shared decoded Wolverines signals

While Michigan tries to navigate the storm around its program, the Wolverines are also trying to make one thing clear: they're hardly alone in trying to
decipher opponents' signals.

Michigan has shared evidence with the Big Ten and NCAA showing staff members at Ohio State, Purdue and Rutgers shared intelligence on the Wolverines' signals
prior to last year's Michigan-Purdue Big Ten Championship Game, The Athletic reported.

Documents breaking down Michigan's signals for numerous plays were compiled by Purdue staffers ahead of the game based on information provided by Ohio State and
Rutgers football personnel, the report says. The person who supplied the documents told The Athletic he spent 10 to 12 hours per week trying to decode signals
based on game video and said the practice of sharing decoded signals with other schools is common.
</item 3>

<item 4>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Within it, the school warned Big Ten commissioner Tony Pettiti against levying "premature" disciplinary action against Jim Harbaugh and alleged that in-person
scouting and sign-stealing are commonplace across the NCAA.

"The conference should act cautiously when setting precedent given the reality that in-person scouting, collusion among opponents, and other questionable
practices may well be far more prevalent than believed," the letter read.

Michigan also included with the letter "two attached documents of Michigan's offensive and defensive signs allegedly stolen and shared by other Big Ten teams,"
per Yahoo Sports.

Conference awaits Michigan response

The Big Ten expects a formal response from Michigan to the notice of discipline by the end of the day Wednesday, ESPN's Pete Thamel reports. His colleague Adam
Rittenberg adds that a ruling is expected to come down Thursday.

Sources: The Big Ten expects a response from Michigan on the notice of discipline by the end of day on Wednesday. Don’t expect any ruling by the Big Ten on
Wednesday, as they are expected to take time to absorb Michigan’s response.
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
However, the primary difference is Michigan is accused of using in-person scouting, which has been prohibited under NCAA rules for nearly 30 years, to acquire
its information.

Monday, Nov. 6

No connection between Buckeyes coach, Michigan investigation

Ohio State coach Ryan Day and his family have no connection to the Big Ten and NCAA investigations into Michigan, Yahoo Sports reported.

A report emerged over the weekend indicating Day may have had some involvement in the probe into the Buckeyes' arch-rivals, but the NCAA told the Big Ten there
are no ties that it knows of between the Ohio State coach and the investigation.

NCAA shares findings with Big Ten as Harbaugh suspension looms

Yahoo Sports reported Monday that the NCAA shared its investigation findings with the Big Ten, noting that the findings "do not connect the in-person scouting
and recording of opponents’ sidelines to Harbaugh."

Still, the conference is considering a suspension for Harbaugh and could hand down punishments before the end of the week.
</item 6>

<item 7>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Helmet technology

CBSSports.com reported on Oct. 20 that the NCAA rules committee is giving teams the option of coach-to-player helmet communication to relay plays from the
sideline during bowl season this year. Tablets will inevitably follow, and here is the sliver of good that will come from this scandal.

It's about damn time. Death to sandwich-board playcalling, please. Bring on those playsheets where coaches simply have to cover their mouth. This is one
instance where emulating the NFL game is overdue.

Sherrone Moore

If Moore is going to be a head coach, then Saturday’s victory against the Nittany Lions was a fitting audition. The Wolverines rushed for 227 yards – including
32 straight run plays – in grinding out a top-10 victory at one of the toughest environments in the FBS at Beaver Stadium.

Moore was emotional - complete with NSFW language – in the aftermath.

Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.
</item 7>

<item 8>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Although the Big Ten has closed its investigation, we are continuing to cooperate with the NCAA as it moves forward with its ongoing investigation. Consistent
with our commitment to integrity, we will continue to take the appropriate actions, including disciplinary measures, based on information we obtain. Earlier
today, Michigan Athletics relieved Chris Partridge of his duties as a member of the Michigan Football staff. Due to employee privacy laws, we are unable to
comment further."

Because Michigan and Harbaugh on Thursday dropped their litigation against the Big Ten, he will not be on Michigan's sideline this week against Maryland and
next week against Ohio State to end the regular season.

Former Wolverines assistant and Harbaugh confidant Biff Poggi on Friday offered his former boss an invitation. Poggi, in his first year as Charlotte's head
coach, asked Harbaugh to join the 49ers for their games vs. Rice on Saturday and at South Florida on Nov. 25.
</item 9>

<item 10>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
2 now, and the debate with the Buckeyes and No. 1 Georgia is going to perk up more than ever. They are the three best teams in the country, but only one has
uncertainty with the head coach.

Whether they admit it or not, Michigan still needs Harbaugh for those moments, especially in two weeks.

The reality is he might not be present for that one either.
</item 10>
=== END ITEM id=6fca2a41bd8a2f0d ===

=== ITEM id=1c3e1d2e228921cf ===
Gold sentences and facts:
Sentence 0: While his Republican colleagues have largely shunned TikTok, Ramaswamy is presenting himself as one of the Cool Politicians who will actually use it.
  Fact 0.0: Ramaswamy's Republican colleagues have largely shunned TikTok.
  Fact 0.1: Ramaswamy is presenting himself as one of the Cool Politicians.
  Fact 0.2: The Cool Politicians are politicians who will actually use TikTok.
Sentence 1: Artists have spent the last few years trying to drum up engagement for their work by making their songs trend on TikTok, which audiences have started to resist.
  Fact 1.0: Artists have spent the last few years trying to drum up engagement for their work by making their songs trend on TikTok.
  Fact 1.1: Audiences have started to resist artists making their songs trend on TikTok.

Retrieved items:
<item 1>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“Gen Z does not put up with bullshit,” Henry said. “That’s so much of what platforms like TikTok and Twitch cater to, and why they thrive on those platforms,
because you can just pull out your phone and be talking while you’re getting ready with the worst angle possible. Republicans in general, it goes against so
much of how they act otherwise and young people know that.”

Ramaswamy’s TikTok strategy is puzzling; he appears to be trying to replicate the success of previous candidates who became social media stars over the course
of their campaigns, but his TikTok presence conflicts with his own stances on social media and young voters. His communications director, Tricia McLaughlin, did
not immediately respond to TechCrunch’s request for comment.

Like many of his Republican colleagues, Ramaswamy has accused TikTok of being a threat to national security, and during a town hall days before he launched his
account, described the platform as “digital fentanyl” from China.
</item 1>

<item 2>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
As the 2024 election approaches, Republican candidates may again fall behind in mobilizing voters on social media platforms.

Ramaswamy is also the only Republican presidential candidate to engage with TikTok, and is one of the few in his party to even have an account. His Silicon
Valley roots as a biotech entrepreneur and millennial upbringing set him apart from the other candidates, but he faces an uphill battle in convincing Gen Z to
like him, much less show up at the polls. Though his videos receive hundreds of thousands of views, his comments are also overrun with sex jokes.

Annie Wu Henry, the digital strategist behind Fetterman’s TikTok stardom, noted that younger voters who are very online tend to value raw, unfiltered
authenticity. They don’t see that in many older politicians, and especially not in the Republican party.

“Gen Z does not put up with bullshit,” Henry said.
</item 2>

<item 3>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“The amount of media and content that we consume, political and cultural, where people get their information … It’s getting more and more intertwined,” Henry
said. “To be effective ideologically, when it comes to campaigns and these wins that we want, we have to understand how these ecosystems work, and be
strategically using them or else we risk being left behind.”

Engaging with voters online requires more than just working with popular creators. In Ramaswamy’s case, an endorsement from Jake Paul only drove animosity from
TikTok viewers. While his older brother, Logan Paul, has made the occasional social commentary on his podcast, the younger Paul has rarely engaged with politics
or social issues. Jake Paul is generally disliked online for the running list of allegations against him, including sexual assault accusations.

The Paul brothers have a reputation of promoting anyone and anything for a paycheck, and when Jake Paul posted a political endorsement out of the blue, viewers
immediately questioned how much Ramaswamy paid him.
</item 3>

<item 4>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Only the occasional tweets. Meet @Vivek Ramaswamy ♬ original sound – GenosPicks

Establishing himself as the millennial politician who’s cool enough to use TikTok but anti-woke enough to play in the divisive culture war isn’t working in
Ramaswamy’s favor, though.

His account, which has amassed over 50,000 followers in the weeks since he joined, has been barraged with comments either criticizing his positions or trolling
him. He’s been the butt of Gen Z’s relentless comments about getting off to his content (the top comments on his videos are consistently jokes about edging). He
also incited the wrath of witchtok creators, who filmed themselves casting hexes upon him and other conservatives.

Ramaswamy is raking in TikTok engagement — even if it’s not how he intended — but Democrats are still dominating the Republican party in digital strategy.
Ocasio-Cortez’s Instagram Stories garnered her nationwide popularity when she campaigned in 2018, reaching further than her local constituency.
</item 4>

<item 5>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
“Luminate’s findings prove what we had long known: that TikTok is the driving force behind music discovery in the industry, and that TikTok users are active,
engaged and highly valuable drivers of music industry revenues.”

The new study is part of an ongoing effort by TikTok to highlight its growing role in the development of global music culture. Last year, TikTok released a
report asserting that 13 out of the 14 songs to reach number on the US Billboard Hot 100 in 2022 were driven by viral trends on TikTok.

The data clearly shows why TikTok has focused so heavily on developing music-related services, to the point that one could be forgiven for thinking the company
plans to reinvent itself as a part of the music industry.

Last year, TikTok launched SoundOn, its own music distribution service, which has partnered with Believe-owned distro company TuneCore and Downtown
Holdings-owned B2B tech and services company FUGA.
</item 5>

<item 6>
Gag City is a viral win for Nicki Minaj (TechCrunch, 2023-12-09T02:34:35+00:00)
Nicki Minaj is an artist who’s been embroiled in controversy throughout her career, from posting bad takes about COVID vaccines to defending her husband Kenneth
Petty, a convicted sex offender. She may be a brilliant artist, but her problematic history makes her far from the family-friendly public figure that brands are
more likely to endorse.

But with Gag City, Minaj has brands doing all of her marketing for her. “Pink Friday 2” is an artistic marvel in itself (though I am probably biased), but the
free promotion that it’s been getting as a viral meme is particularly astounding. Artists have spent the last few years trying to drum up engagement for their
work by making their songs trend on TikTok, which audiences have started to resist. Gag City doesn’t bank on being the viral song of the summer to drive
streaming numbers — the bit is removed enough for non-stans to enjoy it, while still revolving around the album it’s promoting.
</item 6>

<item 7>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
A similar pattern was found in the four other countries they study looked at.

The study also found that the TikTok audience can be an effective marketing tool for artists, as TikTok users, in particular, are considerably more likely to
discover and share music than overall social media and short video platform (SFV) users.

In the UK, TikTok users were 108% more likely to discover and share music than other social media users, while in Germany, they were 100% more likely. US TikTok
users were 71% more likely to discover and share music.

“In doing so, TikTok users help both emerging and established artists to reach new audiences,” the Luminate report stated.

The study found that TikTok’s platform appears to be a major driver of the glocalization of music – the trend of local artists, performing in local languages,
being elevated to a regional or global platform.

“TikTok users have a stronger preference for international music than the average music listener,” the report said.
</item 7>

<item 8>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
The attorney general’s press secretary blamed the site’s downfall on activists “hacking the system.”

The presidential election is more than a year away, but it may be too late for Ramaswamy to make any headway on TikTok, where the platform’s young users still
don’t take him seriously. He recently posted a video about getting ready for the next Republican debate, and was hit with yet another wall of edging comments.

Even if candidates like Ramaswamy did everything right — like having a platform that didn’t alienate young voters and working with creators with more favorable
reputations — they’d still represent a party that many Gen Z and millennial voters aren’t aligned with.

“It’s really hard to be effective with a generation when a lot of your policy is attacking them,” Henry said. “If all of your policy is highly unfavorable for
that generation, that’s a hard sell in itself, even if you’re an effective communicator with all the strategy in the world, to sell someone something they don’t
want.”
</item 8>

<item 9>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 9>

<item 10>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Republicans still don’t know how to talk to young voters online Vivek Ramaswamy struggles to gain an edge in the face of TikTok trolls

In an appeal to younger voters, Republican presidential candidate Vivek Ramaswamy — who proposed raising the voting age to 25 — launched his TikTok presence
with an endorsement from Jake Paul, the YouTuber turned boxer who built a content empire by marketing to children.

Ramaswamy is one of the only Republican politicians making an effort to connect with Gen Z and young millennials, a demographic that overwhelmingly supported
Democrats in the midterm elections. Despite the popularity and growing influence of far-right creators online, Republican candidates have historically failed to
engage young voters on social media, if they try at all.

While Democratic politicians like Rep. Alexandria Ocasio-Cortez and Sen. John Fetterman’s social media campaign strategies set the blueprint for politicians
seeking internet fame, Republicans have neglected to adopt the digital fluency required to secure the youth vote.
</item 10>
=== END ITEM id=1c3e1d2e228921cf ===

=== ITEM id=27eff61cba30c1de ===
Gold sentences and facts:
Sentence 0: He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not.
  Fact 0.0: He also believes that after everything on social media becomes perfectly curated for us, our brains will start to crave things that are not.
  Fact 0.1: He believes that social media becomes perfectly curated for us.
Sentence 1: What we have done since then, all the success and failures [matters in our preparation].
  Fact 1.0: What we have done since then, including all the success and failures, matters in our preparation.

Retrieved items:
<item 1>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Does Reddit co-founder Alexis Ohanian have regrets over his creation? The entrepreneur and founder of venture fund Seven Seven Six, speaking at an event at the
University of Virginia, espoused a fairly negative view of the current social media landscape and the creator economy — an environment where he says, everyone
has “been so trained and conditioned now to just get the likes and retweets, get the upvotes…I wish I could be more optimistic about it, but we’re seeing it
play out before our eyes.” Still, he ultimately described himself as a “techno-optimist” — an apparent nod to a16z General Partner Marc Andreessen’s recent
diatribe where he claimed technology could solve all our problems.

“Guilty as charged,” Ohanian said, during his interview with Kara Swisher. “I do think that pendulum will swing back. I think even swifter than we realize.”
</item 1>

<item 2>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Ian Foster, who made two changes to his starting side and three in total, was forced to defend his centre’s actions while twisting the knife into Sexton’s
finish.

“It happens on most sporting parks when tensions are high and a lot at stake,” Foster said.

“Unfortunately recently it’s also included players saying a few things to referees. Is it right for the game? I don’t know. You don’t hear a lot of players
complaining about it.”

Advertisement

Captain Sam Cane added: “It’s players’ different personalities and emotions running high in those moments. There’s always people who have different tactics but
it’s always just left out there.”

Foster recalled Sam Whitelock to the starting side in the only change to the pack.

Whitelock, whose breakdown penalty win after 37 phases denied Ireland a match-winning penalty, has been recalled and will partner Scott Barrett in the
second-row. Brodie Retallick will come off the bench.
</item 2>

<item 3>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
I’ve got the experience of being in difficult situations before,” Jones said.

“This is not an unusual situation. You’ve got a team that was struggling for a period of time, you’ve got underneath it a system that’s not supporting it and at
some stage you bottom out.

“This is the opportunity now to change the team- as we’ve started with picking younger players with bright futures – and then we’ve got to look at the
development system underneath as well.

“It’s not a simple equation that the Wallabies are bad. Let’s be serious about this. The Wallabies are not where they need to be but underneath that we need to
fix the system to go forward. And if we can do that now we’ve got a rosy future.”

Sports opinion delivered daily
</item 3>

<item 4>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
“Firstly, you’ve never heard us say we’re favourites,” he said.

‘We know these games are do or die. We’re going to have to improve our performance. The stage gets bigger at this time in the tournament and you’ve got to grow
your game. That’s our strong focus. The best team on the night wins – Argentina has done that to us.”

Cheika however was happy to remind the All Blacks who were favourites.

“”New Zealand has always been an example in rugby, a benchmark. They make you think of high-level skills in a really open game but there are always threats in
the lineout, mauls and rucks,” said Cheika, whose Wallabies lost in the 2015 final.

“You have to take the moments you are in the game and see them coming at you at different times.

Advertisement

“In the game in 2020 there were things you could pinpoint.
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
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“To inspire is not about results, it’s not just one player who played well; no, it’s about our behaviour in general, our ability to be ready to come back from
difficult times and seizing our opportunities.

“There are not a lot of opportunities in life – us playing in a semi-final is even rarer and we want to have no regrets.”
</item 6>

<item 7>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
But as for social media as a whole, of which Reddit would presumably play a part, Ohanian said it brings out the “worst parts of our nature in many ways.”
What’s more, he expressed worries about the desire for people to be first and win on whatever leaderboard a platform features, which then leads to a situation
where misinformation is able to spread.

Humans evolved trusting the faces & voices we knew around the campfire. The idea of believing something bc of experts/institutions you've never met is actually
new to our species. I visited @UVADemocracy + chatted with @karaswisher + @dkroy about what's next for social media. pic.twitter.com/PMBXi7T3vv — Alexis Ohanian
🇦🇲 (@alexisohanian) October 31, 2023

“Now, there’s so much pressure to be first, and UGC — user-generated content — will always win. It won’t always be right, though.
</item 7>

<item 8>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 8>

<item 9>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Despite the problems with social media and the urgency, in terms of its impact on society, Ohanian believes there’s still some hope. He noted, for example, that
the founders his fund 776 invests in are much smarter and more aware of the issues than he ever was at their age. Perhaps the next platforms will still be
addictive, but in a way that doesn’t “further disrupt democracy,” he mused.

He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not. “I think we’ll
see a rebirth of live gatherings and in-person experiences…the bright side of this is that I hope it then gets us back in touch with our humanity, and the
things that no amount of AI can replace.”

“I want to be an optimist about it,” Ohanian added. “And I think there’s still a path forward but we’re fighting against very, very strong human things.
</item 9>

<item 10>
Dropout’s dropping CollegeHumor on its 5th anniversary — here’s why and what the future holds (Polygon, 2023-09-26T17:35:00+00:00)
But we’ve given them names and voices. And now we’ll see! Now we’ll see how glib they are!

Reich: David Attenborough couldn’t do this!

Sam, that reminds me of something you said to NPR earlier this year, about how you found success with Dropout in, instead of going broad for the viral video,
going for “something that feels special to a small group of people.” Has five years of Dropout inspired any advice for creators, or producers, or fans —
basically anyone who’s looking to get away from algorithmically powered viral demand, to other platform structures?

Reich: My advice is find a better business model. I think we live in an era of online content right now that is being defined and shaped by algorithms. And
that’s true no matter the platform. I think different platforms are shaping content in different ways, and that’s kind of exciting as a creator. Like, Which of
these sports am I most excited to play?
</item 10>
=== END ITEM id=27eff61cba30c1de ===

=== ITEM id=3b8156da7259bc81 ===
Gold sentences and facts:
Sentence 0: Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it.
  Fact 0.0: Valve has announced a new Steam Deck.
  Fact 0.1: The announcement of the new Steam Deck was a surprise.
  Fact 0.2: The writing outlet has already reviewed the new Steam Deck.
Sentence 1: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 1.0: Valve made a number of minor physical upgrades with its new Steam Deck iteration.
  Fact 1.1: The minor physical upgrades were made alongside big improvements.
Sentence 2: For now, Griffais explained, it's important to Valve that developers don't need to "split their attention" between different performance targets.
  Fact 2.0: Griffais explained that it is important to Valve that developers do not need to "split their attention" between different performance targets.
  Fact 2.1: Griffais said this is important to Valve for now.
Sentence 3: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 3.0: The Steam Deck OLED will go on sale on November 16 at 1pm ET.
  Fact 3.1: Valve has Steam Deck OLED units ready to ship on November 16.

Retrieved items:
<item 1>
WIRED, Polygon, and Engadget’s 2023 deal coverage tracked Apple/Samsung holiday and Prime Day discounts across Macs, iPads, watches, AirPods, and accessories.
The $4,000 16-inch M3 Max MacBook Pro impressed with power, ports, four-display support, XDR screen, and 26h45 battery; M3 Pro suits creators, M3 offers value,
and M2 Airs suit everyday use. iPad deals included ninth-gen at $
</item 1>

<item 2>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Not Sonos fast there: Audio company Sonos scored a big $32.5 million win against Google a while back. Now it transpires that the company’s patents were deemed
unenforceable and invalid. Whoops. A federal judge threw out the $32.5 million win this week.

When it clicks, it really clicks: We took a deep look at Pixel 8, and our team discovered that it delivers solid performance and design upgrades. The camera got
a particularly enthusiastic shout-out, with Darrell declaring that Google’s Pixel 8 Pro camera is the new mobile photography champ, and Brian waxing lyrical
about the phone’s picture-snapper in “The camera’s still the thing.”

Who needs computers anyway?: It seems like all our mobile devices may be starting to cannibalize sales of personal computers — Ron reports that PC shipments
decline slows in Q3 2023 and that Apple’s sales plunge over 23%.

Let’s get together

Reddit’s API-powered chaos continues, but it appears that things are starting to resolve a little.
</item 2>

<item 3>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The focus of the M3 chips is enhanced graphical capabilities. Compared to the previous M1 chip, Apple asserts that the M3 renders at 2.5 times the speed and has
a CPU that is 30% faster. The event’s videos looked great (of course they did — they were produced by Apple), but one delightful quirk was that they were all
shot on iPhones, showing off that the phones are becoming more and more capable, even for professional work.

Magic Leap, an augmented reality hardware company, has replaced its CEO Peggy Johnson with Ross Rosenberg, marking the completion of its enterprise pivot.
Johnson joined Magic Leap in August 2020 and led the shift in business strategy. Her successor previously served as executive at Belden and First Solar, and
will now guide the company through its next period, as the company faces the daunting challenge of achieving profitability.

More from hardware land:

Is it a bird? Is it a plane?: Shield AI, a startup specializing in autonomous flying systems for the U.S.
</item 3>

<item 4>
Engadget’s Nov. 19 Black Friday roundup highlights discounted Apple tablets and laptops. The iPad Air remains a strong choice for performance, battery life,
slim design and Apple Pencil 2 support, though a new M-series lineup was expected early next year. The 10th-gen iPad’s 64GB model is $349 at Amazon, Target and
Best Buy—$100 off, its lowest price—with a 10.9-inch display. Lenovo’s IdeaPad Flex 5i Chromebook
</item 4>

<item 5>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 5>

<item 6>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 6>

<item 7>
2023 WIRED, Engadget, Polygon, and Verge guides covered price-checked deals on TVs, audio, phones, wearables, gaming, smart-home gear, appliances, baby and pet
products, and services. Picks included LG/Samsung OLEDs, Sonos, Switch OLED, Quest 2, Pixel, Apple Watch, Nest displays, cameras, and kitchen appliances. Gaming
roundups compared affordable Xbox, premium Elite/Scuf, and 8BitDo/PDP controllers, plus console
</item 7>

<item 8>
Engadget’s Nov. 19, 2023 Black Friday roundup notes an Apple tablet’s front camera has moved to the landscape edge, improving video calls, but it still supports
only the first-generation Apple Pencil. Apple’s M2 Mac mini is discounted to $500 for 256GB at Best Buy and $690 for 512GB at Amazon with a clippable coupon.
Despite an expected M3 upgrade, the current model remains a speedy macOS desktop; Engadget rated it 86
</item 8>

<item 9>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
GPU updates were the centerpiece of the news for several reasons. One is a feature the company calls Dynamic Caching. Apple writes,

With Dynamic Caching, only the exact amount of memory needed is used for each task. This is an industry first, transparent to developers, and the cornerstone of
the new GPU architecture. It dramatically increases the average utilization of the GPU, which significantly increases performance for the most demanding pro
apps and games.

More simply put, the idea is allocating just the right amount of local memory to a given task. It’s something Apple has been working on for a while. The other
two additions should be more familiar for anyone who’s spent time behind a game controller. The first is ray tracing, a feature designed to bring added realism
to the gaming experience, though more accurate depictions of lighting and reflections.

That feature is now hardware-accelerated, along with mesh shading. The triangle computer shader makes processing complex geometrical images more efficient.
</item 9>

<item 10>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
There were still a few during the event, but four-year-old non-proprietary chips are no longer the main point of comparison. Enough time and upgrades have
elapsed since the M1 arrived. In fact, this Friday marks three years since it was announced. Apple again,

Rendering speeds are now up to 2.5x faster than on the M1 family of chips. The CPU performance cores and efficiency cores are 30 percent and 50 percent faster
than those in M1, respectively, and the Neural Engine is 60 percent faster than the Neural Engine in the M1 family of chips.

It’s worth pointing out here that Apple’s chip numbering system isn’t as straight forward as it seems at first blush. For instance, while the M3 represents an
advance over the M2, you can still expect to see, say, the M2 Ultra to outperform the M3. Pros, Maxes and Ultras are progressively larger versions of their
baseline chip, containing more CPU, GPU and neural engine cores.
</item 10>
=== END ITEM id=3b8156da7259bc81 ===

=== ITEM id=a1a5f8651be19108 ===
Gold sentences and facts:
Sentence 0: “I’m here until the end,” she said in her press conference on Friday.
  Fact 0.0: A female speaker said 'I'm here until the end' in her press conference on Friday.
Sentence 1: He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with the London club went so poorly.
  Fact 1.0: A man has been out of work since his tenure with the Blues came to a close back in April.
  Fact 1.1: It would be a hard sell to fans for this man to return, since his time with the London club went so poorly.

Retrieved items:
<item 1>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
A representative from her camp declined to comment for this story.

Crocker said in September that the final interviews would include lengthy technical and tactical assessments, as well as questions to determine the candidates’
cultural fit. He and the federation stayed fairly consistent on their desired start date since the head coach role opened in August, but that became one of the
major concessions made by U.S. Soccer in selecting Hayes.

She’s headed Stateside 🇺🇸 U.S. Soccer has appointed Emma Hayes as the 10th full-time head coach in #USWNT history » https://t.co/sR7qfSnWVx Welcome, Emma!
pic.twitter.com/WYn6Sg9RmX — U.S. Women's National Soccer Team (@USWNT) November 14, 2023

Hayes will remain exclusively with Chelsea through the end of their WSL campaign and the Champions League season. She will not work with the U.S. in
international windows.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.
</item 3>

<item 4>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
The 44-year-old Italian has led the tiny club to new heights, qualifying for the Europa League for the first time in club history and establishing themselves as
top-four contenders, building upon the success under Graham Potter. Yet could Potter's failure at Chelsea cause some trepidation around another Brighton success
story moving on up?

Michael Carrick

This list wouldn't be complete without mention of the likeliest of those with direct Manchester United links. Former United midfielder Michael Carrick, who made
316 appearances for the club over a 12-year Old Trafford tenure, is at Championship club Middlesbrough, his first permanent managerial appointment. They sit
11th in the second tier as of mid-December, just three points back of the playoff spots in a crowded top half.

Carrick's lack of experience is glaring, but hiring someone with deep knowledge of the club could be very enticing, and the fact he hasn't bombed in his first
job is notable.
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 6>

<item 7>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
It is a global context." Does he still feel the same way?

Graham Potter

While Potter's time at Chelsea crashed and burned, many still rate the former Brighton & Hove Albion boss highly and he could be in line for a project job such
as this. Indeed, there have previously been rumours that Sir Jim Ratcliffe, the INEOS founder on the brink of purchasing a 25 per cent stake in United, would
like Potter at the club. He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with
the London club went so poorly.

Roberto De Zerbi

One of the fastest rising managers in the game, Brighton manager Roberto De Zerbi is known for his tactical innovation in transitional play, and his name has
cropped up a number of times when big clubs have an opening.
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.
</item 10>
=== END ITEM id=a1a5f8651be19108 ===

=== ITEM id=ecf4615e9294589f ===
Gold sentences and facts:
Sentence 0: Here's your chance to claim $5,000+ in bonuses as a new user from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, and BetRivers.
  Fact 0.0: The reader has a chance to claim $5,000+ in bonuses as a new user.
  Fact 0.1: A new user can claim bonuses from Caesars Sportsbook.
  Fact 0.2: A new user can claim bonuses from BetMGM.
  Fact 0.3: A new user can claim bonuses from Bet365.
  Fact 0.4: A new user can claim bonuses from FanDuel.
  Fact 0.5: A new user can claim bonuses from DraftKings.
  Fact 0.6: A new user can claim bonuses from BetRivers.
Sentence 1: Collect your welcome bonus: As a new customer, you’ll receive a welcome bonus.
  Fact 1.0: The reader can collect a welcome bonus.
  Fact 1.1: A new customer receives a welcome bonus.

Retrieved items:
<item 1>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
into your FanDuel account and place a qualifying . Whether your bet wins or loses, you receive $200 in Bonus Bets .

. You can use your Bonus Bets in whatever increments you like, but you have to use your Bonus Bets within seven days or they will expire.

or they will expire. Winning wagers with Bonus Bets award the winnings only and not the bet amounts.
</item 1>

<item 2>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
That means wagering requirements are attached, and payouts don't include the stake.

Odds boosts

You’ll receive better than the usual odds on specific bets. Daily odds boosts are available, but provide no value if the boosted bet isn't a winner.

Bet insurance

Get a refund if you lose a specific bet or your parlay comes up one leg short. Sportsbooks will limit the size of the refund you can claim, and most refunds are
bonus bets or site credit, not cash. That means wagering requirements are attached, and payouts don't include the stake.

Bet & get bonuses

Here, you’ll get a bonus when you place a specific bet. Most bet & get bonuses are bonus bets or site credit. That means wagering requirements are attached, and
payouts don't include the stake.

App-only offers

Place a specific bet with one of the best sports betting apps to receive a bonus. This won’t be available on the sportsbook website.
</item 2>

<item 3>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.

Claim here: Bet365 bonus code TSNCO for promo of your choice

FanDuel promo code for Bet any $5 moneyline, Get $150 in bonus bets if your team wins in CBK game

FanDuel has a new offer when using our exclusive FanDuel promo code link. Bet $5 on any moneyline, and Get $150 in bonus bets if your team wins!

It's important to note that there are no odds restrictions, so it'd be prudent to bet a big favorite over an outmatched underdog on the CBK schedule. You need
to win the wager to receive $150 in bonus bets.
</item 3>

<item 4>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Winning wagers with Bonus Bets award the winnings only and not the bet amounts.

FanDuel Sportsbook welcome offer pros and cons

Pros

Unlike other offers, you don’t have to risk much of your own money to receive the bonus — just $5 to get $200 in Bonus Bets

Also unlike other offers, it doesn’t matter if your initial wager wins or loses, you get the bonus either way

You don’t have to use your Bonus Bets all at once (unless you want to), as you can divide up how you use the Bonus Bets any way you like

Cons

You can’t just withdraw the Bonus Bets, but you have to wager them

You only have a week to use all $200 of your Bonus Bets

If you win with Bonus Bets, you only get the winnings and not the wager amounts back

What is the welcome offer at FanDuel Sportsbook VT?

Those signing up at FanDuel Sportsbook Vermont by Jan.
</item 4>

<item 5>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking on our exclusive DraftKings promo code link

Register as a new user by supplying personal info including your name, email, last four of your SSN#, and mailing address

Deposit $5 to activate $50 bonus bet

Bet $5 and Get $150 in bonus bets

To max out DraftKings' deposit match at $1,000, make a minimum deposit of $5,000 and meet the 25X playthrough requirement.

Claim here: DraftKings promo code for $1,200 in bonus bets

BetRivers promo code SPORTSRIV unlocks 2nd chance bet for CBK games including Duke vs. Arkansas

The BetRivers promo code SPORTSRIV gives new users a 2nd chance bet up to $500 depending on your state.

If your first bet settles as a loss, BetRivers will refund your account in bonus bets.
</item 5>

<item 6>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
The promo codes on this page get you the best bonuses available. When you plug in the code during registration, the sportsbook knows we sent you and gives you
the bonus we negotiated for you.

Are these promo codes for regular customers, too?

The promo codes on this page are for new customers only. Once you’ve got an account with a sportsbook, you can take advantage of other bonus offers and ongoing
promos.

Related Pages:
</item 6>

<item 7>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
DraftKings | BetMGM vs. FanDuel | Caesars vs. BetMGM
</item 7>

<item 8>
College Bowl Odds, Spreads, Money Lines and Totals - NCAAF 2023 Bowl Season (Sporting News, 2023-12-15T20:37:00+00:00)
It’s a quick process. Below, we outline the steps for opening your own account.

Choose your sportsbook: If you’re not sure which sportsbook is the right fit for you, browse our reviews to see the pros and cons of each, as well as the
current offerings for new customers. When you find the sportsbook that best suits your needs, simply click on a link from our site to visit the homepage of your
chosen sportsbook. Register for an account: On the homepage, you’ll find a button that says “sign up” or “register.” Once you click that, you’ll be taken to a
form for providing personal information such as your name, birth date, email address, and the last four digits of your Social Security number. You’ll also be
required to accept the sportsbook terms and conditions, which we recommend you read. Collect your welcome bonus: As a new customer, you’ll receive a welcome
bonus.
</item 8>

<item 9>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Here are more details on the offer explaining how each part of the DraftKings Sportsbook welcome offer works:

Sign up for a DraftKings Sportsbook account via the links on this page .

. Make a first deposit of at least $5 and get a $50 Bonus Bet .

and get a . When you make your first deposit, DraftKings will give you a 20% deposit match bonus worth up to $1,000 in Bonus Bets . Be aware this bonus comes
with a 25x wagering requirement .

worth up to . Be aware this bonus comes with a . Finally, when you place an initial bet of at least $5 , you get $150 more in Bonus Bets delivered as eight $25
Bonus Bets that you’ll have seven days to use.

, you get delivered as eight $25 Bonus Bets that you’ll have seven days to use. If you go for the full deposit match bonus, you can receive as much as $1,200 in
Bonus Bets all told.
</item 9>

<item 10>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
This offer requires a minimum $10 deposit, but you can deposit up to the amount of your first bet to $1,500 to maximize your potential winnings. If your first
bet on the app loses, BetMGM will refund your account with bonus bets in the amount of the original wager up to $1,500.

The extra $50 bonus bet offer is not available in Washington D.C., Kentucky, Mississippi, Nevada, New York, Puerto Rico and Ontario. Customers in those areas
can use BetMGM bonus code SPORTSPICK for the $1,500 First Bet offer.
</item 10>
=== END ITEM id=ecf4615e9294589f ===
