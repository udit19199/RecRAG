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

=== ITEM id=115b593811f76570 ===
Gold sentences and facts:
Sentence 0: “I’m here until the end,” she said in her press conference on Friday.
  Fact 0.0: A female speaker said 'I'm here until the end' in her press conference on Friday.
Sentence 1: He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with the London club went so poorly.
  Fact 1.0: A man has been out of work since his tenure with the Blues came to a close back in April.
  Fact 1.1: It would be a hard sell to fans for this man to return, since his time with the London club went so poorly.

Retrieved items:
<item 1>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
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
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 6>

<item 7>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Advertisement

The hiring process included multiple rounds of evaluation, with the list of candidates becoming smaller each time. The first pass was driven purely by data,
which was then whittled down to a double-digit list Crocker was considering as of September, and then a final shortlist, which also included Tony Gustavsson,
head coach of Australia. Multiple sources confirmed both Hayes and Gustavsson flew to the U.S. for interviews.

One source who was briefed on the situation said the federation had also checked in on the availability of Sarina Wiegman, despite clear messaging from both the
English FA and Wiegman herself in August. “I’m staying out of it. I’ve heard it (from the press officer) but no, I’m with England and I’m really happy with
England, and I have a contract until 2025,” Wiegman said. A representative from her camp declined to comment for this story.
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
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
A representative from her camp declined to comment for this story.

Crocker said in September that the final interviews would include lengthy technical and tactical assessments, as well as questions to determine the candidates’
cultural fit. He and the federation stayed fairly consistent on their desired start date since the head coach role opened in August, but that became one of the
major concessions made by U.S. Soccer in selecting Hayes.

She’s headed Stateside 🇺🇸 U.S. Soccer has appointed Emma Hayes as the 10th full-time head coach in #USWNT history » https://t.co/sR7qfSnWVx Welcome, Emma!
pic.twitter.com/WYn6Sg9RmX — U.S. Women's National Soccer Team (@USWNT) November 14, 2023

Hayes will remain exclusively with Chelsea through the end of their WSL campaign and the Champions League season. She will not work with the U.S. in
international windows.
</item 10>
=== END ITEM id=115b593811f76570 ===

=== ITEM id=91c0c8e445fb51e6 ===
Gold sentences and facts:
Sentence 0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has been more than just a curiosity.
  Fact 0.0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme.
  Fact 0.1: Inside the Big Ten, the alleged Michigan sign-stealing scheme has been more than just a curiosity.
Sentence 1: While it is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough stain to erase knowing those violations are now on the Big Ten record.
  Fact 1.0: It is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons.
  Fact 1.1: The in-person scouting and sign-stealing are alleged violations.
  Fact 1.2: The alleged violations are now on the Big Ten record.
  Fact 1.3: Knowing the alleged violations are on the Big Ten record will be a tough stain to erase.

Retrieved items:
<item 1>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
We (the Big Ten head coaches) want something done now. I don’t think people understand the advantage that what they’re (allegedly) doing gives you. People
think, ‘OK, now that everybody knows, we all can just move on.’ Like, ‘now, it’s fair.’ Well, no, it isn’t. Not at all. This changes the way you operate. A lot
of teams have been doing things a certain way for years. Now, it’s forcing you to teach your players a whole new way to communicate just for them. People think
that this is just advanced scouting. This was damn near espionage.”

A third Big Ten head coach told The Athletic that this is “one of the most egregious breaches in the spirit of the game” he’s ever heard of.

“They (Michigan) have been manipulating the game and cheating the game for two-and-a-half years.
</item 1>

<item 2>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
To know exactly what the other team is doing, Michigan might as well have been playing with 15 guys on the field,” he said. “What’s the message the Big Ten is
sending now by doing nothing? Win now, pay later? We might as well just send people to (scout) their practices and their games. It doesn’t encourage anybody to
follow the rules. It’s just telling them to do the opposite and say, f— it.”

GO DEEPER Michigan should be punished, say 94% of CFB coaches in our poll. What else did it reveal?

The NCAA is investigating Michigan’s football program amid allegations that the Wolverines used illegal in-person scouting and the recording of signals to steal
signs this season. Ahead of the Michigan-Michigan State game on Oct. 21, the Big Ten approached MSU and said it was made aware of “credible evidence” regarding
the sign-stealing allegations. The Big Ten said it would monitor the NCAA’s investigation into Michigan.
</item 2>

<item 3>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.
</item 3>

<item 4>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
“They’re reading the tea leaves and wondering why the Big Ten hasn’t done anything yet. Every week and every day that goes by, people are like, ‘Something’s
gotta give.’ It’s getting a little bit out of hand when you see him (allegedly) on the Central Michigan sideline. The playing field is not level right now. How
can you have a team that you know has a competitive advantage over you still being allowed to play? That’s what the coaches are grappling with.”

Advertisement

“It feels like (former commissioner) Kevin (Warren) taking over and then COVID,” the source continued. “Tony’s walking into this situation, and people are
calling for the league to make a statement before they have all the facts.”

Despite frustration from all corners of the conference, sources at four different Big Ten schools said they do not expect the conference to levy any sort of
punishment against Michigan before the season ends.
</item 4>

<item 5>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten does have the ability to act under its sportsmanship policy, but that doesn’t mean that it would want to act quickly or decisively before the NCAA
completes its entire investigation and allows Michigan a chance to respond to its findings.

This is an unprecedented situation; whatever Petitti decides to do (or not do) will set a precedent. The Big Ten itself doesn’t have investigators, so it needs
to rely upon the NCAA to do that part — and to determine who else was involved in the alleged scouting scheme. It’s not clear exactly what the coaches would
want the league to do to punish Michigan; banning the team from competing in the Big Ten championship, for example, would harm players who had nothing to do
with the sign-stealing apparatus.

A source briefed on the coaches’ call said Big Ten coaches are concerned about whether Michigan “should represent the Big Ten.”

“No matter what happens, if Michigan continues to move forward, the clouds will follow,” the source said.
</item 5>

<item 6>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has
been more than just a curiosity. On Wednesday’s Big Ten coaches video call with commissioner Tony Petitti, and after Jim Harbaugh left the call, that
frustration was voiced loud and clear, according to conference coaches, who said they don’t feel like the new Big Ten commissioner is “motivated” to do anything
about the Wolverines.

Advertisement

“There is just a ton of frustration,” a Big Ten coach told The Athletic on Thursday morning. “Look at Jim Harbaugh’s record before this started. The guy was on
the hot seat before 2021, and now he’s like the king of college football. … No doubt this all has had a profound effect.
</item 6>

<item 7>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten said it would monitor the NCAA’s investigation into Michigan.

Advertisement

“The Big Ten is so much more powerful than the NCAA,” that third Big Ten coach said. “Why are you just sitting back and doing nothing about this? The Big Ten
can’t pound its chest for the last 30 years about how it does the right thing ethically (when other conferences like the SEC won’t) and then have this go on. If
this were a team in the bottom half of the Big Ten, would this be handled in the same way?

“When a running back gets hurt against Michigan because they knew exactly what play was coming, will that kid and his family have the ability to sue the Big
Ten?”

The NCAA investigation is ongoing, a process that typically moves slowly, which makes it difficult to imagine it will reach a resolution by the time the
postseason begins.
</item 7>

<item 8>
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
</item 8>

<item 9>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
ESPN reported that Big Ten coaches and athletic directors encouraged Big Ten commissioner Petitti to act last week, but that resulted in action before the NCAA
investigation concluded.

Do you wonder what the SEC and commissioner Greg Sankey privately think about all this? For the Big Ten, it is an ugly run of bad behavior on and off the field
for a conference that is welcoming four new members next season in Oregon, UCLA, USC and Washington.

To be determined

Michigan

Michigan is 10-0 and on track to meet No. 1 Ohio State for another blockbuster matchup in college football’s best rivalry. On the field, the Wolverines remain
on track for a third straight Big Ten championship and shot at the College Football Playoff.

But this is about more than football.

Will more penalties come from the Big Ten and NCAA as a result of the in-person scouting scandal that has not left the college football consciousness this
month? Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan?
</item 9>

<item 10>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
… No doubt this all has had a profound effect.

“This guy’s being investigated for three different things now between the (alleged) illegal signal stealing, the (alleged) illegal recruiting during COVID and
that investigation into the offensive coordinator and alleged computer hacking. There are guys (on that call) who could lose jobs, and then there’s this guy
over here (Harbaugh) who is gonna get a new, bigger contract now, and they won’t do anything about him.”

Asked to describe the tone of the coaches’ sentiment expressed to Petitti, another Big Ten coach called it “angry” — particularly at the Big Ten’s lack of
action, or even apparent interest in taking any.

“Everybody’s upset,” that coach told The Athletic. “Why is nothing being done? We want to know, what else do you need to know to take action? We (the Big Ten
head coaches) want something done now.
</item 10>
=== END ITEM id=91c0c8e445fb51e6 ===

=== ITEM id=38492619d67b4b8a ===
Gold sentences and facts:
Sentence 0: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 0.0: The case will attempt to argue that Google restrains competition within two separate markets.
  Fact 0.1: One of the two markets is the distribution of apps to Android users.
  Fact 0.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 1: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 1.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.
  Fact 1.1: The three categories are relevance, distance, and prominence.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 2>

<item 3>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 3>

<item 4>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 10>
=== END ITEM id=38492619d67b4b8a ===

=== ITEM id=bfd95cb159dd5e03 ===
Gold sentences and facts:
Sentence 0: A number of videos posted to X since the attacks have been identified as entirely unrelated to the conflict — including footage that was filmed last month in Egypt and even a clip from a video game that had been posted to the platform with a (false) claim it showed Hamas missile attacks on Israel.
  Fact 0.0: A number of videos posted to X since the attacks have been identified as entirely unrelated to the conflict.
  Fact 0.1: One unrelated video was filmed last month in Egypt.
  Fact 0.2: Another unrelated item was a clip from a video game.
  Fact 0.3: The video game clip was posted to X with a false claim that it showed Hamas missile attacks on Israel.
Sentence 1: We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.
  Fact 1.0: The outlet reached out to the Commission with questions.
  Fact 1.1: The questions ask how the Commission plans to respond to this latest analysis of X's role in spreading disinformation.

Retrieved items:
<item 1>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
X should have “proportionate and effective mitigation measures” in place to tackle “the risks to public security and civic discourse stemming from
disinformation,” he also says.

Instead, the platform appears to be turning into an engine of disinformation — one that’s demonstrably enabled the very swift amplification of a smorgasbord of
toxic fakes around the Israel-Hamas war. Falsehoods that may be trying to manipulate opinion around the conflict, or otherwise exploit horrific events to drive
engagement (clickbait) or for even more bleak, cynical and potentially harmful ends.

“Public media and civil society organisations widely report instances of fake and manipulated images and facts circulating on your platform in the EU, such as
repurposed old images of unrelated armed conflicts or military footage that actually originated from video games. This appears to be manifestly false or
misleading information,” writes Breton. “I therefore invite you to urgently ensure that your systems are effective, and report on the crisis measures taken to
my team.”
</item 1>

<item 2>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“I remind you that following the opening of a potential investigation and a finding of non-compliance, penalties can be imposed,” Breton adds, concluding the
letter.

We’ve asked the Commission to confirm whether it has opened an investigation into X’s DSA compliance over concerns it’s raised in the letter. Perhaps it’s going
to wait a day to see what his response is before taking that next dance step.

Musk’s extremely iterative — or just plain erratic/arbitrary — management style is so far from the responsive and responsible qualities the EU’s rulebook
demands of digital leaders it’s hard to see how this clash can end well for either side.

We contacted X for a response to the EU’s warnings about its DSA compliance but at press time the company had not responded — beyond firing back its usual
automated reply, which reads: “Busy now, please check back later.”
</item 2>

<item 3>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” he warns. “We have, from qualified sources, reports about potentially illegal content circulating on your service despite flags from relevant
authorities.”

He also takes issue with a change X made last night to its public interest policy, under which it judges newsworthiness (i.e., for deciding whether posts that
infringe its rules may nonetheless remain on the site) — but which Breton says have left “many European users uncertain” (i.e., about how X is applying its own
rules).

Again, this is a problem because the DSA requires platforms to be clear and transparent about their rules and how they apply them. “This is particularly
relevant when it comes to violent and terrorist content that appears to circulate on your platform,” Breton continues in another pointed warning.
</item 3>

<item 4>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
In a very swift test of the European Union’s newly updated content moderation rulebook, the bloc has fired a public warning at Elon Musk–owned X (formerly
Twitter) for failing to tackle illegal content circulating on the platform in the wake of Saturday’s deadly attacks on Israel by Hamas terrorists based in the
Gaza Strip.

The European Commission has also raised concerns about the spread of disinformation on X related to the terrorist attacks and their aftermath.

Unlike terrorism content, disinformation is not illegal in the EU per se. However, the EU’s Digital Services Act (DSA) puts an obligation on X — as a so-called
very large online platform — to mitigate risks attached to harmful falsehoods as well as to act diligently on reports of illegal content.

Following the terrorist attacks by Hamas against 🇮🇱, we have indications of X/Twitter being used to disseminate illegal content & disinformation in the EU.
</item 4>

<item 5>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.
</item 5>

<item 6>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.
</item 6>

<item 7>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
A rule of law clash between Musk and the EU has looked increasingly inevitable in recent months — and, indeed, has been predicted by some industry watchers
since rumors of the erratic billionaire’s plan to take over Twitter emerged last year.

Reminder: Penalties for confirmed breaches of the DSA can be as high as 6% of global annual turnover. The bloc also has powers, in extremis, which could result
in access to X being shut off in the region if the platform repeatedly fails to correct course. So if Musk fails to satisfy EU regulators, he could face serious
consequences for what remains a highly indebted business.

In his letter to Musk, Breton writes that the EU has “indications” that X is being used to disseminate illegal content and disinformation in the EU” following
Saturday’s attacks. He then goes on to remind the company of the DSA’s “very precise obligations” vis-à-vis content moderation.
</item 7>

<item 8>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
He’s also ripped up a bunch of legacy content moderation policies and slashed in-house enforcement teams while promoting a decentralized, crowdsourced
alternative (rebranded as Community Notes), which essentially outsources responsibility for dealing with tricky issues like disinformation to users in what
looks suspiciously like another gambit to eke out extra engagement and farm confusion by applying a philosophy of extreme relativism so culture warriors are
encouraged to keep forever fighting for their own “truth” in the comments.

Oh and he also pulled X out of the EU’s Code of Practice on Disinformation earlier this year in a very clear thumb of the nose to EU regulators.

Urgent letter to Musk

In an “urgent” letter to Musk today, which the EU’s internal market commissioner Thierry Breton also shared on X, the bloc has sent the strongest signal yet it
believes Musk’s platform is in breach of the DSA — although this is by no means Musk’s first warning.
</item 8>

<item 9>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
Asking Musk to be “effective” against disinformation is a bit like asking the sea to stop moving. But, well, this is how the regulatory dance must go (and after
the dance comes the denouement — which, if a DSA breach is confirmed, means enforcement. And actual penalties might be a bit harder for Musk to troll).

In the meantime, the EU has asked Musk to make contact with relevant law enforcement authorities and Europol — and “ensure that you respond promptly to their
requests.” Breton also flags some unspecified additional DSA compliance issues he says his team will be contacting Musk’s about “shortly” — “with a specific
request.”. (We’ve asked the EU what else it’s concerned about on X and will update this report with any response.)

The bloc has given Musk a deadline of 24 hours to respond to its asks at this point — stipulating his answer will be added to its assessment file on X’s
compliance with the DSA.
</item 9>

<item 10>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
EU turns up the heat on X over illegal content in wake of Israel-Hamas war

The European Union is cranking up the heat on Elon Musk-owned X. Late Thursday, the Commission sent the company a formal request for more information — after
issuing a public warning Wednesday about reports that illegal content and disinformation targeting the Israel-Hamas war is circulating on the platform, and
Twitter’s CEO sending a high-level (non specific) response to that warning.

The move could prefigure the opening of a formal investigation of X under the bloc’s content moderation rulebook, the Digital Services Act (DSA). If the
Commission moves ahead on that, it will be the first investigation opened under the DSA since a compliance deadline for so called “very large online platforms”
(aka VLOPs) kicked in this summer.

Non-compliance under DSA rules, meanwhile, could trigger fines of up to 6% of annual turnover, plus potential blocking of a service for repeated infringements.
X owner Elon Musk last projected revenues of about $3 billion for this year.
</item 10>
=== END ITEM id=bfd95cb159dd5e03 ===

=== ITEM id=2c63d5f2bba35365 ===
Gold sentences and facts:
Sentence 0: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 0.0: Miami has five remaining opponents.
  Fact 0.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.
Sentence 1: 152.67 -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: The sentence begins with the number 152.67.
  Fact 1.1: Tyreek Hill missed last week's game.
  Fact 1.2: Tyreek Hill now needs to average almost 153 yards per game in his final three games.
  Fact 1.3: Tyreek Hill's stated goal is 2,000 receiving yards this season.

Retrieved items:
<item 1>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 1>

<item 2>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
9:43 p.m. — Hill is up to 65 yards on eight catches. No other Dolphins receiver has more than two catches, and Hill now has 879 yards on the season.

Eagles 17, Dolphins 10

9:39 p.m. TOUCHDOWN — Tagovailoa hits Hill in the end zone for a touchdown! Hill was facing double coverage but simply beat both defenders. Miami has the
deficit back to one possession.

9:38 p.m. — Rookie Nolan Smith sacks Tagovailoa with 44 seconds remaining in the half.

9:37 p.m. — The Dolphins are down to the Eagles' 29-yard line after catches by Braxton Berrios and Tyreek Hill. They have a chance to score on back-to-back
possessions, as they get the ball out of halftime.

9:33 p.m. TWO-MINUTE WARNING — Woah!
</item 2>

<item 3>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs DAL MIA -1.5 O/U 51 OPP VS WR 7th PROJ PTS 14.2 WR RNK 16th YTD Stats REC 71 TAR 99 REYDS 964 TD 4 FPTS/G 14.8 We expect Hill to
return this week, which should drive people off of Waddle. But there is at least a chance that Hill won't be 100% and Waddle will still be Tua Tagovailoa's top
target. Waddle has been the quintessential DFS tournament play as of late with three of seven games over 19 PPR Fantasy points, topping 26 twice.

WR Preview Heath's Projections

My full set of Week 16 Fantasy Football projections for every position are now available on SportsLine. Find out which of my favorite plays are projected to
score higher than consensus rankings and which don't live up to their draft hype, at least in Week 16.
</item 3>

<item 4>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
MORE SUPERDRAFT DFS & FANTASY PROPS: How to play SuperDraft

This one should be a tightly-contested, high-scoring game — with heavy playoff implications, because we all know this could be an AFC Championship preview — so
expect Rice to be very busy through all four frames in Frankfurt. The Dolphins have allowed the fifth-most receptions and the 12th-most yards to wideouts this
season. If a bounce-back week is meant to be, Sir Rashee is our destiny.

Austin Ekeler, RB, Chargers @ Jets — OVER 29.5 receiving yards

Guess who's back? Back again...Ek is back....tell a friend. The Chargers finally got their best player back to full health, with Ekeler exploding for 123 yards
and a TD against the Bears last weekend. A whopping 94 of those 123 yards came through the air, as the PPR god gashed Chicago for chunk yardage a handful of
times.
</item 4>

<item 5>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 5>

<item 6>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 7>

<item 8>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 8>

<item 9>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 9>

<item 10>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The Commanders offensive line has had some ugly showings lately against the Cowboys (four sacks, 49% pass rush pressure rate allowed), the Giants (four sacks,
32.7% pressure rate allowed) and the Seahawks (three sacks, 46.8% pressure rate allowed). Somehow these results are better than what they were allowing through
much of the first two months of the year. Sam Howell getting the ball out faster has been a positive in this regard but a negative for his numbers in three of
his past four games (the one game he did really well in was sort of fluky). The matchup is a good test for the Dolphins pass rush post-Phillips, one that it
should pass without giving up much in the way of deep completions.

STARTS: Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2),
</item 10>
=== END ITEM id=2c63d5f2bba35365 ===
