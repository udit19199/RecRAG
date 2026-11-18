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

This file has 3 items.

=== ITEM id=1697d45207689053 ===
Gold sentences and facts:
Sentence 0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has been more than just a curiosity.
  Fact 0.0: Coaches across college football have been riveted by the alleged Michigan sign-stealing scheme for the past two weeks.
  Fact 0.1: Inside the Big Ten, the alleged Michigan sign-stealing scheme has been more than just a curiosity.
Sentence 1: While it is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough stain to erase knowing those violations are now on the Big Ten record.
  Fact 1.0: It is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons.
  Fact 1.1: The alleged in-person scouting and sign-stealing violations are now on the Big Ten record.
  Fact 1.2: Having the alleged violations on the Big Ten record will be a tough stain to erase.

Retrieved items:
<item 1>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
To know exactly what the other team is doing, Michigan might as well have been playing with 15 guys on the field,” he said. “What’s the message the Big Ten is
sending now by doing nothing? Win now, pay later? We might as well just send people to (scout) their practices and their games. It doesn’t encourage anybody to
follow the rules. It’s just telling them to do the opposite and say, f— it.”

GO DEEPER Michigan should be punished, say 94% of CFB coaches in our poll. What else did it reveal?

The NCAA is investigating Michigan’s football program amid allegations that the Wolverines used illegal in-person scouting and the recording of signals to steal
signs this season. Ahead of the Michigan-Michigan State game on Oct. 21, the Big Ten approached MSU and said it was made aware of “credible evidence” regarding
the sign-stealing allegations. The Big Ten said it would monitor the NCAA’s investigation into Michigan.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Michigan, Big Ten, Michigan State, Michigan-Michigan State game, sign-stealing allegations, illegal in-person scouting, recording
of signals, NCAA, Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 1>

<item 2>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
… No doubt this all has had a profound effect.

“This guy’s being investigated for three different things now between the (alleged) illegal signal stealing, the (alleged) illegal recruiting during COVID and
that investigation into the offensive coordinator and alleged computer hacking. There are guys (on that call) who could lose jobs, and then there’s this guy
over here (Harbaugh) who is gonna get a new, bigger contract now, and they won’t do anything about him.”

Asked to describe the tone of the coaches’ sentiment expressed to Petitti, another Big Ten coach called it “angry” — particularly at the Big Ten’s lack of
action, or even apparent interest in taking any.

“Everybody’s upset,” that coach told The Athletic. “Why is nothing being done? We want to know, what else do you need to know to take action? We (the Big Ten
head coaches) want something done now.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Michigan, Big Ten, Petitti, Harbaugh, alleged illegal signal stealing, alleged illegal recruiting during COVID,
alleged computer hacking, investigation into the offensive coordinator, Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 2>

<item 3>
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

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Michigan, Big Ten, SEC, sign-stealing, NCAA investigation, NCAA, Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near
Espionage’
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

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Michigan, Big Ten, Covid, sign-stealing, Kevin Warren, Tony, Central Michigan, Michigan’s Rivals Fume About Sign-Stealing: ‘This
Was Damn Near Espionage’
</item 4>

<item 5>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
We (the Big Ten head coaches) want something done now. I don’t think people understand the advantage that what they’re (allegedly) doing gives you. People
think, ‘OK, now that everybody knows, we all can just move on.’ Like, ‘now, it’s fair.’ Well, no, it isn’t. Not at all. This changes the way you operate. A lot
of teams have been doing things a certain way for years. Now, it’s forcing you to teach your players a whole new way to communicate just for them. People think
that this is just advanced scouting. This was damn near espionage.”

A third Big Ten head coach told The Athletic that this is “one of the most egregious breaches in the spirit of the game” he’s ever heard of.

“They (Michigan) have been manipulating the game and cheating the game for two-and-a-half years.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Michigan, Big Ten head coaches, A third Big Ten head coach, sign-stealing, cheating, espionage, advanced scouting,
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 5>

<item 6>
Graph query row:
article: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti
connection: PUBLISHED_BY
related_entity: Sporting News
portrayal_context: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
The Big Ten announced Michigan coach Jim Harbaugh will be prohibited from coaching on the sideline for the remainder of the regular season.

At least, in theory. The Friday ruling is the next chapter in an NCAA investigation into Michigan for in-person scouting and sign stealing in a scandal that led
to the resignation of staffer Connor Stallions on Nov. 3. Harbaugh will be allowed to coach during the week but will not be able to be on the sideline. Michigan
did not get a ruling on a temporary restraining order ahead of the showdown against No. 10 Penn State.

The Wolverines still won 24-15 under interim coach Sherrone Moore, but the fallout from the scandal continues to have a far-reaching impact on the Big Ten,
Michigan and college football beyond this year's College Football Playoff race.

A look at the winners and losers from the scandal to date:

Michigan scandal winners and losers

Winners

Tony Petitti

Petitti certainly did not hit Michigan with a "slap on the wrist."
</item 6>

<item 7>
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

Graph facts:
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, college football, College Football Playoff, Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti, Greg
Sankey, Big Ten, in-person scouting scandal, ESPN, USC, Jim Harbaugh, Ohio State, UCLA, Sporting News, Washington, SEC
</item 7>

<item 8>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has
been more than just a curiosity. On Wednesday’s Big Ten coaches video call with commissioner Tony Petitti, and after Jim Harbaugh left the call, that
frustration was voiced loud and clear, according to conference coaches, who said they don’t feel like the new Big Ten commissioner is “motivated” to do anything
about the Wolverines.

Advertisement

“There is just a ton of frustration,” a Big Ten coach told The Athletic on Thursday morning. “Look at Jim Harbaugh’s record before this started. The guy was on
the hot seat before 2021, and now he’s like the king of college football. … No doubt this all has had a profound effect.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Michigan, college football, Big Ten, Jim Harbaugh, Tony Petitti, sign-stealing scheme, Big Ten coaches video call,
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’
</item 8>

<item 9>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
Stalions bought tickets to games in at least seven Big Ten stadiums before those teams played the Wolverines over the past three seasons, including the 2023
season, sources told The Athletic last month. Purchasing the tickets is not a violation of NCAA rules, but using them to scout and record other teams would
violate the rules, prohibiting in-person, on-campus scouting and the audio or video recording of signals.

“They aren’t allegations. It happened,” Purdue coach Ryan Walters said Thursday night on his radio show ahead of Saturday’s game against the Wolverines.
“There’s video evidence. There’s ticket purchases and sales you can track back. We know for a fact they were at a number of our games. We’ve had to teach our
guys a new language.”

On Monday, coach Jim Harbaugh met with reporters and said “the people that know us the most think the most of us” as Michigan faces the NCAA investigation. He
has denied knowledge of the alleged scouting.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Michigan, Big Ten, Stalions, Jim Harbaugh, sign-stealing, NCAA, Michigan’s Rivals Fume About Sign-Stealing: ‘This
Was Damn Near Espionage’, Ryan Walters, Purdue, NCAA investigation, In-person, on-campus scouting, Audio or video recording of signals
</item 9>

<item 10>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten does have the ability to act under its sportsmanship policy, but that doesn’t mean that it would want to act quickly or decisively before the NCAA
completes its entire investigation and allows Michigan a chance to respond to its findings.

This is an unprecedented situation; whatever Petitti decides to do (or not do) will set a precedent. The Big Ten itself doesn’t have investigators, so it needs
to rely upon the NCAA to do that part — and to determine who else was involved in the alleged scouting scheme. It’s not clear exactly what the coaches would
want the league to do to punish Michigan; banning the team from competing in the Big Ten championship, for example, would harm players who had nothing to do
with the sign-stealing apparatus.

A source briefed on the coaches’ call said Big Ten coaches are concerned about whether Michigan “should represent the Big Ten.”

“No matter what happens, if Michigan continues to move forward, the clouds will follow,” the source said.
</item 10>
=== END ITEM id=1697d45207689053 ===

=== ITEM id=d1e7dbe1cad4b4a2 ===
Gold sentences and facts:
Sentence 0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 0.0: Epic will suggest that Google hid items from discovery by allowing execs and employees to have "off the record" chats.
  Fact 0.1: Google will also counter Epic's suggestion that Google hid items from discovery.
  Fact 0.2: Google has provided the court with thousands of chat logs and millions of documents.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: She admits that she may have misinterpreted the evidence.
  Fact 1.1: She maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content, their readers and ad revenue through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Publishers -PARTICIPATED_IN-> class action antitrust suit
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal

Entities: class action antitrust suit, The Wall Street Journal, deal to license news for AI model training, News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line, antitrust, The Atlantic, Axel Springer, TechCrunch, AI, Similarweb, Google, OpenAI
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- Alphabet -ACCUSED_OF-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.

Entities: Sherman Act, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, U.S. Justice Department,
Hausfeld, Helena World Chronicle, LLC, class action antitrust suit against Google, U.S. Justice Department lawsuit against Google, 2020 Justice Department civil
antitrust suit over search and search advertising, digital ad technologies, Clayton Act, search and search advertising, digital news and reference publishing,
class-wide monetary and injunctive relief, free marketplace of ideas, free press
</item 2>

<item 3>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> monopoly power
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Bing

Entities: monopoly power, FTC-Facebook case, Bing, Apple, WhatsApp, TechCrunch, data, Google antitrust case, 5 things we learned so far about the Google
antitrust case, DOJ, FTC, FTC lawsuit against Amazon, anticompetitive and unfair strategies, DOJ investigation of Apple, Apple's policy for third-party apps on
its devices
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Helena World Chronicle, U.S. District Court in D.C., Search Generative Experience, Bard AI chatbot, Knowledge Graph, anticompetitive
behavior, U.S. antitrust law, Sherman Act, Arkansas, question-and-answer technologies, News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line, class action lawsuit, TechCrunch, AI technologies
</item 5>

<item 6>
Graph query row:
article: 200+ of the best deals from Amazon's Cyber Monday sale
published_on: 2023-11-27T08:45:59+00:00
relationship: MENTIONS
connected_entity: Mashable's daily Deals newsletter
</item 6>

<item 7>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Wired opinion article, Megan Gray, Is Google Search better than the rest? And is that fair?, Walker, The Age, US, Google, trial, WIRED
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.

Graph facts:
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn -MENTIONS-> Publishers
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet

Entities: Knowledge Graph, Publishers, Knowledge Panel, Featured Snippets, class action antitrust suit, misappropriation of Publishers’ content, The Wall Street
Journal, report on Thursday, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, antitrust, TechCrunch, AI,
Google
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.
- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
-MENTIONS-> Washington Post
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Search Generative Experience

Entities: SGE, Google Bard AI, GoogleBot, Google general search service, AdSense, News Media Alliance, Washington Post, Epic Games lawsuit against Google over
app store antitrust issues, app store antitrust issues, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line,
class action antitrust suit against Google, Epic Games, Allen Institute for AI, TechCrunch, Google
</item 10>
=== END ITEM id=d1e7dbe1cad4b4a2 ===

=== ITEM id=5b393cf506a6d727 ===
Gold sentences and facts:
Sentence 0: Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio headsets on Delta flights.
  Fact 0.0: Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals on Delta flights.
  Fact 0.1: Cardholders can get 20% back as a statement credit when they purchase eligible in-flight food purchases on Delta flights.
  Fact 0.2: Cardholders can get 20% back as a statement credit when they purchase eligible alcoholic beverages on Delta flights.
  Fact 0.3: Cardholders can get 20% back as a statement credit when they purchase eligible audio headsets on Delta flights.
Sentence 1: And for Black Friday, it's on sale for over $400 off, knocking the price down to a super impressive $297.
  Fact 1.0: For Black Friday, the item is on sale for over $400 off.
  Fact 1.1: For Black Friday, the item's price is knocked down to $297.

Retrieved items:
<item 1>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases, alcoholic beverages, and audio
headsets on Delta flights. Baggage insurance. Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance
coverage—which kicks in if your baggage is stolen, lost, or damaged.

Cardholders who purchase plane, train, ship or bus ride tickets with their card are eligible for baggage insurance coverage—which kicks in if your baggage is
stolen, lost, or damaged. Car Rental Loss and Damage Insurance. A cardholder can get damage and theft insurance on their car rental. Cardholders must use their
card to book the rental and decline the collision damage waiver at the agency to receive it.
</item 1>

<item 2>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Our take: If you’re a small business owner who flies with Delta regularly and wants to score a solid welcome bonus, consider the Delta SkyMiles® Gold Business
American Express Card. This card has a welcome bonus of up to 75,000 miles and perks like free checked bags and priority boarding with a $0 intro annual fee for
the first year (then $99).

Pros Free checked bag free on Delta flights

Generous welcome bonus of 75,000 miles

No foreign transaction fees Cons No lounge access

$99 annual fee ($0 intro fee for the first year)

To view rates and fees of the Delta SkyMiles® Gold Business American Express Card, see this page.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, Delta SkyMiles® Gold Business American Express Card, Delta, Fly with Delta frequently? Small business owners can score a
solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card
</item 2>

<item 3>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta SkyMiles Gold Business Card Highlights

Card Type: Airline

Welcome bonus: 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your first 6 months

75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your first 6 months Annual fee: $0 intro annual fee for the first year (then $99)

$0 intro annual fee for the first year (then $99) Rewards: 2x miles on purchases made directly with Delta, 2x miles on U.S. purchases for advertising in select
media, 2x miles on U.S. shipping purchases, 2x miles on dining at restaurants, 1x mile on every eligible dollar spent on other purchases

With the Delta SkyMiles® Gold Business American Express Card, cardholders who frequently fly with Delta can reduce travel costs and upgrade their travel
experience.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, Delta, Fly with Delta frequently?, Delta SkyMiles Gold Business Card
</item 3>

<item 4>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta has more than 20 airline partners, so cardholders can book through airlines like Air France and Virgin Atlantic—though the 15% discount only applies to
Delta and Delta Connection® flights.

This discount is a reason to hold onto the card long-term as you could save thousands of miles over the course of a year if you are a heavy Delta user. The best
bargains on award flights are found through Delta SkyMiles Award Deals. Recent deals from Seattle (SEA) have included Chicago (ORD) for 8,000 miles or Auckland,
New Zealand (AKL) for 64,000 one way in economy plus taxes and fees. Flash sales come and go so it’s a good idea to keep an eye on the site.

Other options include booking travel through Delta Vacations, using miles for upgrades or even buying champagne at the Delta Sky Club. That said, your best
value will most likely be on award tickets.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chicago
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> Chicago
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Chicago
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> Chicago
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> Chicago

Entities: Chicago, Fortune, Auckland, Seattle, New Zealand, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other
perks with the Delta SkyMiles Gold Business Card, Air France, Virgin Atlantic, Delta Connection, Delta SkyMiles Award Deals, Delta Vacations, Delta Sky Club,
Delta SkyMiles Gold Business Card
</item 4>

<item 5>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
For $99 a year, cardholders receive benefits like priority boarding, the first checked bag benefit (and for guests), discounts on in-flight purchases, and
baggage insurance. The card, however, doesn’t provide access to Delta Sky Club or Delta’s airport lounges.

The rewards structure on this card is decent, but it’s nothing to write home about—the card offers 2x miles on just a handful of categories like advertising,
shipping, and dining.

A big benefit of the Delta SkyMiles Gold Business Card is the welcome bonus: 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your
first 6 months of card opening. And cardholders who use their miles to purchase flights get even more benefits—customers receive a 15% discount when they use
miles to buy Delta flights on the Fly Delta app or the Delta website.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Delta Sky Club, Fly Delta app, Delta website, Fly with Delta frequently? Small business owners can score a solid welcome bonus and
other perks with the Delta SkyMiles Gold Business Card, Delta SkyMiles Gold Business Card
</item 5>

<item 6>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Credit cards similar to Delta SkyMiles Gold Business Card

Cardholders who aren’t loyal to Delta and want more flexible options for redeeming their rewards should consider the Chase Ink Business Preferred® Credit Card,
which has a similar annual fee but a more generous welcome bonus and rewards structure.

Ink Business Preferred credit card vs. Delta SkyMiles Gold Business Card

The Chase Ink Business Preferred® Credit Card has an annual fee of $95 and a large welcome bonus of 100,000 points after spending $8,000 in the first three
months of card opening.

With the Ink Business Preferred Card, customers receive 3x points per $1 (up to $150,000 combined) spent on shipping purchases, internet, cable, phone, and
advertising with social media sites and search engines. Ultimate Rewards points can be transferred 1:1 to Virgin Atlantic or Flying Blue to book flights on
Delta.

Is the Delta SkyMiles Gold Business Card right for you?

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business
Card, Virgin Atlantic, Delta SkyMiles Gold Business Card, Chase Ink Business Preferred® Credit Card, Chase, Ultimate Rewards, Flying Blue
</item 6>

<item 7>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
That said, your best value will most likely be on award tickets.

Rates and fees

Annual fee: $0 intro annual fee for the first year (then $99)

$0 intro annual fee for the first year (then $99) Foreign transaction fee: None

None Purchase APR: 20.99%–29.99% variable

Additional benefits

Free first checked bag. Cardholders (and up to nine of their guests) can get the first checked bag for free. The standard bag fee is $30, so if you use this
perk more than three times, you’ll save more in bag fees than the annual fee costs you.

Cardholders (and up to nine of their guests) can get the first checked bag for free. The standard bag fee is $30, so if you use this perk more than three times,
you’ll save more in bag fees than the annual fee costs you. Main Cabin 1 Priority boarding.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business
Card, Delta SkyMiles Gold Business Card
</item 7>

<item 8>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Delta SkyMiles Gold Business Card: Rewards

The Delta SkyMiles® Gold Business American Express Card provides higher rewards rates on select purchases that might be helpful to small business owners:

2x miles on purchases made directly with Delta

2x miles on U.S. purchases for advertising in select media

2x miles on U.S. shipping purchases

2x miles on dining at restaurants

1x mile on every eligible dollar spent on other purchases

New cardholders can receive a welcome bonus of 75,000 Bonus Miles after you spend $6,000 in purchases on your new Card in your first 6 months of card opening.

How to redeem Delta SkyMiles Gold Business Card miles

Purchase flights

Cardholders can use their miles (and their cash) to buy flights. When cardholders book flights through the Fly Delta app or Delta SkyMiles account with their
cards, they can get a 15% discount called Takeoff15.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, U.S., Delta, Fly Delta app, Delta SkyMiles account, Takeoff15, Fly with Delta frequently? Small business owners can score a
solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card, Delta SkyMiles Gold Business Card
</item 8>

<item 9>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Is Delta SkyMiles Gold Card hard to get?

Cardholders should aim for a ‘good’ FICO credit score of 690 or better to qualify for the Delta SkyMiles Gold Card, though card issuers look at other factors
like income, too.

Please note that card details are accurate as of the publish date, but are subject to change at any time at the discretion of the issuer. Please contact the
card issuer to verify rates, fees, and benefits before applying.

Terms apply to American Express benefits and offers. Enrollment may be required for select American Express benefits and offers. Visit americanexpress.com to
learn more.

Eligibility and Benefit level varies by Card. Terms, Conditions, and Limitations Apply. Please visit americanexpress.com/benefitsguide for more details.
Underwritten by Amex Assurance Company.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, American Express, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta
SkyMiles Gold Business Card, Delta SkyMiles Gold Business Card, Delta SkyMiles Gold Card, FICO credit score, Amex Assurance Company, americanexpress.com,
americanexpress.com/benefitsguide, income, rates, fees, and benefits, American Express benefits and offers
</item 9>

<item 10>
Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business Card (Fortune,
2023-12-12T21:01:43+00:00)
Main Cabin 1 Priority boarding. Cardholders (and up to nine guests on the reservation) can get priority boarding if they use their SkyMiles number when booking
their Delta flights.

Cardholders (and up to nine guests on the reservation) can get priority boarding if they use their SkyMiles number when booking their Delta flights. Flight
credit. If cardholders make eligible purchases totaling $10,000 or more in a year, they will receive a $100 Delta eCredit that can be used to purchase Delta
flights.

If cardholders make eligible purchases totaling $10,000 or more in a year, they will receive a $100 Delta eCredit that can be used to purchase Delta flights.
20% back on in-flight purchases. Cardholders can get 20% back as a statement credit when they purchase eligible pre-purchased meals, in-flight food purchases,
alcoholic beverages, and audio headsets on Delta flights.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Delta, Fly with Delta frequently? Small business owners can score a solid welcome bonus and other perks with the Delta SkyMiles Gold Business
Card, SkyMiles, Delta eCredit, Delta flights, priority boarding, flight credit, in-flight purchases, Delta SkyMiles Gold Business Card
</item 10>
=== END ITEM id=5b393cf506a6d727 ===
