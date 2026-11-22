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

=== ITEM id=f9a5bdb04daf5f3f ===
Gold sentences and facts:
Sentence 0: Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to keep their hands off of some social media posts.
  Fact 0.0: The Supreme Court decided last week that it would hear the pair of cases.
  Fact 0.1: The pair of cases revolve around Republicans crafting state-specific laws.
  Fact 0.2: The state-specific laws order platforms to keep their hands off of some social media posts.
Sentence 1: Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation.
  Fact 1.0: Under Musk, Twitter has prioritized free speech.
  Fact 1.1: Twitter is now called X.
  Fact 1.2: Under Musk, free speech on Twitter is managed by crowdsourced-based moderation.

Retrieved items:
<item 1>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.
</item 1>

<item 2>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What’s at stake in the Supreme Court’s landmark social media case The court will wade into First Amendment questions with major implications for platforms

The Supreme Court is poised to decide whether a pair of state laws are allowed to reshape the ability of social media companies to control what does — and
doesn’t — appear on their platforms.

Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to
keep their hands off of some social media posts. Since the early days of the Trump administration, Republicans have accused social media companies of
deliberately suppressing conservative viewpoints.

While research has not supported these claims, researchers have demonstrated that conservative social media users are disproportionately exposed to political
misinformation, a phenomenon that could explain anecdotal claims of ideologically lopsided enforcement on social platforms.

Whether animated by those perceived differences or the political perks of accusing social platforms of anti-conservative bias, conservative lawmakers in Florida
and Texas passed laws to restrict how those companies are allowed to moderate content.
</item 2>

<item 3>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“They’re in the same ballpark, the sentiment is the same,” Barrett said. “Republican lawmakers in each state believe that — and said explicitly in the course of
debating and passing these laws — that ‘Silicon Valley oligarchs’… are ideological liberals, and they are censoring people in our states who are conservative,
and we are hereby ordering them to stop doing that.”

How could the Supreme Court decision affect social media companies?

If the court finds that social media companies don’t have a First Amendment right to curate the kinds of content that they allow, social platforms could look
very different, at least in states that are trying to limit their moderation powers. After years of slow progress on misinformation — and worrisome backsliding
on platforms like Elon Musk’s X — the Supreme Court’s decision could upend that process, sowing chaos online in the process.
</item 3>

<item 4>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation. Its policies have resulted in increased hate speech
and toxic content on the platform, some studies have shown.

Mozilla believes there’s a better way forward and, more importantly, that now is exactly the right time to invest in that path. And, as a wholly owned
subsidiary of a nonprofit, the company says it’s not motivated by generating earnings for shareholders or returning a VC investment, allowing it to progress
with a collaborative approach where it takes in input from a lot of different voices.

In an interview with TechCrunch, Mozilla Senior Director of Content Carolyn O’Hara explained why Mozilla has taken an interest in the fediverse and Mastodon,
specifically, and how it expects to experiment in the months ahead.
</item 4>

<item 5>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Aside from forcing platforms to allow some forms of content that would otherwise be disallowed, these laws also seek to force social media companies to provide
users with individualized explanations when their content is removed or restricted. Because this process is largely conducted algorithmically now — generally
with light human intervention or oversight — social media companies might need to reimagine their content moderation systems or hire way more humans to respond
to these incidents. Those kind of adjustments would likely be costly and difficult to scale.

“As you can imagine, that can get kind of onerous when you’re taking down millions and millions of pieces of content a day, and much of that activity of the
vast majority of it is currently being done automatically,” Barrett said. “The idea that a human being would have to go back and explain each time something
came down would be quite a challenge.”
</item 5>

<item 6>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Time will tell how this one plays out but he may find the dead blue bird he’s chosen to hang around his neck, when he purposefully ripped up the standard
content moderation rulebook and opted to channel online chaos, starting to feel like it’s dragging him down.
</item 6>

<item 7>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
Ofcom recommends these platforms implement policies like not allowing strangers to send direct messages to children, using hash matching to detect and remove
CSAM, maintaining content and search moderation teams, and offering ways for users to report harmful content.

Large tech platforms already follow many of these practices, but Ofcom hopes to see them implemented more consistently. “We think they represent best practice
of what’s out there, but it’s not necessarily applied across the board,” Whitehead says. “Some firms are applying it sporadically but not necessarily
systematically, and so we think there is a great benefit for a more wholesale, widespread adoption.”

There’s also one big outlier: the platform known as X (formerly Twitter). The UK’s efforts with the legislation long predate Elon Musk’s acquisition of Twitter,
but it was passed as he fired large swaths of its trust and safety teams and presided over a loosening of moderation standards, which could put X at odds with
regulators.
</item 7>

<item 8>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
They must also clearly communicate their T&Cs to users and be able to demonstrate proper enforcement.

Larger platforms like X also have additional obligations to identify and mitigate systemic risks such as disinformation, gender-based violence or negative
effects on the exercise of fundamental rights.

Additionally, the regulation includes a “crisis response” mechanism (Article 36) which enables the Commission to adopt rapid-fire measures on larger platforms
in situations of “serious threat”, such as war.

Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show
Hamas missile attacks on Israel but was actually footage from a video game.

Meanwhile X’s ability to respond internally to reports of content problems have been drastically pared back after Musk’s takeover last year — which saw major
layoffs, including in content moderation and human rights as part of the new billionaire owner’s bid to improve the platform’s profitability.
</item 8>

<item 9>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 9>

<item 10>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 10>
=== END ITEM id=f9a5bdb04daf5f3f ===

=== ITEM id=04cd30523d08f502 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that “all of that, all of it, was built on lies.”
Sentence 1: FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.
  Fact 1.0: Gary Wang is an FTX co-founder.
  Fact 1.1: Gary Wang is the former CTO of FTX.
  Fact 1.2: Caroline Ellison is the former CEO of Alameda Research.
  Fact 1.3: Gary Wang pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.
  Fact 1.4: Caroline Ellison pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.

Retrieved items:
<item 1>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.
</item 1>

<item 2>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 2>

<item 3>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Ellison is expected to take the stand under a similar arrangement.

U.S. District Judge Lewis Kaplan presides as Gary Wang testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

Born in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to the Massachusetts Institute of Technology to study math and computer
science. He worked at Google after college. Wang, who first met Bankman-Fried during high school at a summer camp, owned 10% of Alameda, while his boss owned
the other 90%. Wang told the court about the advantages that Alameda received by having code baked into FTX's software that allowed special access to the crypto
exchange.
</item 4>

<item 5>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 7>

<item 8>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
As a result of today’s testimony, I am both more and less confused. I am clearer on what probably happened at FTX. What I don’t understand is why Gary Wang did
it.

Wang was the co-founder and chief technology officer of FTX, as well as the co-founder of Alameda Research. He owned only 10 percent of Alameda and 17 percent
of FTX. He got a $200,000 a year salary, with no performance bonuses. (Okay, sure, at times he was a billionaire on paper.) By contrast, Bankman-Fried owned 90
percent of Alameda and 65 percent of FTX.

Yet Wang’s testimony suggests he took bizarre risks for his own relatively small share. Bankman-Fried didn’t code, so none of his fingerprints are on the
technical changes that let FTX allegedly defraud its customers and investors. Meanwhile, Wang supervised the code behind a key prosecution claim: that Alameda
Research was afforded secret, special privileges with FTX’s funds.
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 10>
=== END ITEM id=04cd30523d08f502 ===

=== ITEM id=6e8deca595902935 ===
Gold sentences and facts:
Sentence 0: Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a return to fitness, and has been included on the squad.
  Fact 0.0: Johnny Cardoso had to pull out of the September camp due to an ankle injury.
  Fact 0.1: Johnny Cardoso has returned to action with Brazilian club Internacional.
  Fact 0.2: Johnny Cardoso's return to action indicates a return to fitness.
  Fact 0.3: Johnny Cardoso has been included on the squad.
Sentence 1: He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.
  Fact 1.0: Johnny Cardoso and his fresh legs are back just in time.
  Fact 1.1: Rookie sensation De'Von Achane hit Injured Reserve this week.
  Fact 1.2: De'Von Achane hit Injured Reserve with a knee injury.

Retrieved items:
<item 1>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 1>

<item 2>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 2>

<item 3>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 3>

<item 4>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: Breaking down Germany squad, tactics as Julian Nagelsmann debuts as manager

USA roster for October friendlies vs Germany, Ghana

The USMNT squad for the October friendlies was announced on Thursday, October 5. Gregg Berhalter has called in 23 players, which is one fewer than were brought
in for the friendlies against Uzbekistan and Oman last month.

Below is the official 23-man roster for the upcoming October friendlies against Germany and Ghana.

PSV midfielder Malik Tillman was forced to withdraw from the squad due to injury, with Alex Zendejas of Club America named to the roster in his place.

MORE: Tracking every goal scored by Christian Pulisic at new club AC Milan

Pos Name Club Caps Goals GK Ethan Horvath Nott. Forest (ENG) 9 0 GK Matt Turner Nott. Forest (ENG) 33 0 DEF Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino
Dest PSV Eindhoven
</item 4>

<item 5>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 5>

<item 6>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He featured his usual pressing, and came close to scoring once on a messy chance that Germany just barely managed to clear, but otherwise he didn't quite cause
the opposing back line problems to open spaces up for teammates.

75th min: Johnny Cardoso — 6

Johnny Cardoso only had 15 minutes at the end to showcase his abilities, but it begged the question why he or Lennard Maloney weren't the ones introduced at
halftime instead of Luca de la Torre. Cardoso lost possession just once in his 10 touches and completed a tackle, although he was also dribbled past a few times
as well.

75th min: Kevin Paredes — 5

With 10 touches in his 15 minutes of action towards the end of a mucked up game of numerous substitutions, Kevin Paredes didn't stand out, losing possession
four times and completing four passes. Tough to evaluate his time on the field, but he didn't stand out.
</item 6>

<item 7>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.

DOLPHINS: Aren't quite the bastion of rushing that some headlines have made them out to be, but they're just over league average in rush rate at 43.7% and above
league average in RB runs per game at 22.6. All of these numbers are weighted heavily by their 50-point win over Denver in Week 3 when they ran the ball 42
times with their running backs. Their 7.6 yard per carry average is due to Achane's breakout start more than anything else.
</item 7>

<item 8>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 10>
=== END ITEM id=6e8deca595902935 ===

=== ITEM id=59443e3854711dc5 ===
Gold sentences and facts:
Sentence 0: Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.
  Fact 0.0: Al Saadi said her frustration with Meta is common among Palestinians.
  Fact 0.1: Al Saadi said her frustration with Meta is common among Palestinians in occupied territories.
  Fact 0.2: Al Saadi said her frustration with Meta is common among Palestinians across the diaspora.
Sentence 1: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents.
  Fact 1.0: Meta wants to shift the burden of monitoring teens' social media usage back to the app stores.
  Fact 1.1: Meta wants to shift the burden of monitoring teens' social media usage back to parents.

Retrieved items:
<item 1>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
His argument essentially was that the improvements and safety features Meta was implementing were not enough and not effective.

However, a policy position like the one Meta announced today is not something that would have been rushed out in a matter of days as a PR response to Bejar’s
claims, but something Meta has been stewing on for some time. After all, the app stores already have age guidelines, so it would seem to follow that they should
also enforce them, if enforcement is to be regulated — or that seems to be Meta’s thinking.

If anything, the call to bring the app stores into the fight is a response to the lack of action in Congress to pass online child safety regulations, like the
hotly debated KOSA (the Kids Online Safety Act) leaving U.S. states to individually create laws due to a void of federal guidelines.

Utah, for instance, passed a measure that will require social networks to give Utah parents access to their children’s posts, messages and comments, and would
block social media access during certain hours.
</item 1>

<item 2>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“And so I think that is what is the most problematic right now about that particular rule, and the way that it’s enforced.”

Over the 13 days leading up to the ceasefire between Hamas and Israel, 7amleh documented more than 500 reports of Palestinian “digital rights violations,”
including the removal and restriction of content, hashtags and accounts related to the conflict.

Meta blamed some of the instances of perceived censorship to technical issues, like one that prevented users in Palestine and Colombia from posting Instagram
Stories. It attributed others to human error, like blocking the hashtag for Al-Aqsa Mosque, the holy site where Israeli police clashed with Ramadan worshippers,
because it was mistaken for a terrorist organization. The company also blocked journalists in Gaza from WhatsApp without explanation.

The same month, a group of Facebook employees filed internal complaints accusing the company of bias against Arab and Muslim users. In internal posts obtained
by BuzzFeed News, an employee attributed the bias to “years and years of implementing policies that just don’t scale globally.”
</item 2>

<item 3>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.
</item 3>

<item 4>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“It feeds into the bias against Palestinian narrative genuinely. It really takes the balance against Palestinians in a situation where there’s a huge asymmetry
of power.”

A history of suppression

Content about Palestine is disproportionately scrutinized, as demonstrated during the last severe violent outbreak between Hamas and Israel two years ago. Amid
the violence following the May 2021 court ruling to evict Palestinian families from Sheikh Jarrah, a neighborhood in occupied East Jerusalem, users across
Facebook and Instagram accused Meta of taking down posts and suspending accounts that voiced support for Palestinians.

The digital rights nonprofit Electronic Frontier Foundation (EFF) described Meta’s actions in 2021 as “systemic censorship of Palestinian voices.” In its 2022
report of Palestinian digital rights, 7amleh said that Meta is “still the most restricting company” compared to other social media giants in the extent of its
moderation of the Palestinian digital space.
</item 4>

<item 5>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
During the last violent outbreak between Hamas and Israel in 2021, a report commissioned by Meta and conducted by a third party concluded that the company’s
actions had an “adverse human rights impact” on Palestinian users’ right to freedom of expression and political participation.

Meta communications director Andy Stone declined to comment on the record, and pointed TechCrunch to Meta’s newsroom post about the conflict, which was updated
on Oct. 18 with a statement addressing the shadowbanning claims.

“Our policies are designed to keep people safe on our apps while giving everyone a voice,” the post stated. “We apply these policies equally around the world
and there is no truth to the suggestion that we are deliberately suppressing voice … We can make errors and that is why there is an appeals process for people
to tell us when they think we have made the wrong decision, so we can look into it.”

The belief that Meta shadowbans, or limits the visibility of, content about Palestine is not new.
</item 5>

<item 6>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Most recently, Instagram users accused Meta of “shadowbanning” their Stories about Palestine.

It’s the latest in a lengthy history of incidents on Meta platforms that reflect an inherent bias against Palestinian users in its processes, as documented by
years of complaints from both inside and outside the company. The company may not intentionally suppress specific communities, but its moderation practices
often disproportionately affect Palestinian users.

For instance, Meta struggles to navigate the cultural and linguistic nuances of Arabic, a language with over 25 dialects, and has been criticized for neglecting
to adequately diversify its language resources. The company’s black-and-white policies often preclude it from effectively moderating any nuanced topic, like
content that discusses violence without condoning it. Advocacy groups have also raised concerns that Meta’s partnerships with government agencies, such as the
Israeli Cyber Unit, politically influence the platform’s policy decisions.
</item 6>

<item 7>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.
</item 7>

<item 8>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The governor also signed a bill to prevent social media companies from adding features designed to addict minors to their platforms. A judge in Arkansas
recently blocked a law that would have required parental permission for minors to create social media accounts.

Beyond Meta, Montana also became the first state to ban TikTok, claiming it was being used by the Chinese Communist Party using TikTok to spy on Americans. But
Meta’s threat, in terms of teen usage, isn’t one of a foreign power collecting private data on users, but rather risks to teen mental health, body comparison
issues and exposure to sexual predators. Montana could easily turn its eyes to Meta next, after the TikTok bill went through.

The overhead of managing apps to meet individual state regulations would be a headache for Meta, but Davis also argues it would lead to a lack of teen
protection, in some cases.

“U.S.
</item 8>

<item 9>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
“U.S. states are passing a patchwork of different laws, many of which require teens (of varying ages) to get their parent’s approval to use certain apps, and
for everyone to verify their age to access them,” she wrote. “Teens move interchangeably between many websites and apps, and social media laws that hold
different platforms to different standards in different states will mean teens are inconsistently protected,” Davis said.

Meta’s proposed “better way” of requiring parents to manage a teen’s app downloads pushes the burden back to the app stores and to parents, who often today do
not monitor or manage their teens’ smartphone use. But while it’s true that the app stores and smartphone makers would know the ages of their users, requiring
parents to approve apps is the equivalent of overriding parental discretion and choice.

Other entertainment ratings — like those for TV, movies or video games — are meant to be guidelines for parents, not laws. Meta’s position that apps are somehow
different, then, is a fairly extreme position to take.
</item 9>

<item 10>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.
</item 10>
=== END ITEM id=59443e3854711dc5 ===

=== ITEM id=78a6b718dd4f8643 ===
Gold sentences and facts:
Sentence 0: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 0.0: Judge Lewis Kaplan intervened and asked whether Bankman-Fried had ever been told by Yedidia about that money
  Fact 0.1: The question asked whether Bankman-Fried had been told about that money in words or in substance
  Fact 0.2: Bankman-Fried admitted he had been told about that money
  Fact 0.3: Bankman-Fried admitted being told about that money only after Judge Lewis Kaplan's question
Sentence 1: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry.
  Fact 1.0: Before his fall, Bankman-Fried presented himself as the Good Boy of crypto
  Fact 1.1: Bankman-Fried presented himself as the trustworthy face of a sometimes-shady industry, crypto
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud
  Fact 2.1: The prosecution said the fraud was committed to achieve great wealth, power and influence
  Fact 2.2: The defense argued that the FTX founder, Bankman-Fried, acted in good faith
  Fact 2.3: The defense argued that Bankman-Fried never meant to commit fraud or steal
  Fact 2.4: The defense argued that Bankman-Fried basically got in over his head

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 1>

<item 2>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 2>

<item 3>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 3>

<item 4>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 4>

<item 5>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 7>

<item 8>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”
</item 8>

<item 9>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 9>

<item 10>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 10>
=== END ITEM id=78a6b718dd4f8643 ===
