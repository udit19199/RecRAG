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

=== ITEM id=d940616aad36e17f ===
Gold sentences and facts:
Sentence 0: The Buckeyes settled for a 52-yard field goal late in the first half, which Jayden Fielding missed wide left.
  Fact 0.0: The Buckeyes settled for a 52-yard field goal late in the first half.
  Fact 0.1: Jayden Fielding missed the Buckeyes' 52-yard field goal wide left.
Sentence 1: His 31-yard field goal is good, and the Buffs take the first lead of the game on their first drive.
  Fact 1.0: A 31-yard field goal is good.
  Fact 1.1: The Buffs take the first lead of the game on their first drive.

Retrieved items:
<item 1>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Can the Buffaloes finally connect for a touchdown before its too late?

UCLA adds on another 💪@UCLAFootball makes it 21-9 in the 4th quarter vs Colorado pic.twitter.com/6Eu9gn6kgn — FOX College Football (@CFBONFOX) October 29, 2023

End of third quarter: Colorado 9, UCLA 14

10:20 p.m. – What a catch for the Bruin! After review, Garbers pass to Loya is ruled complete.

Colorado 9, UCLA 14

10:06 p.m. FIELD GOAL – Mata nails a 34-yard field goal to lessen Colorado's deficit to five. Four minutes remain in the third quarter.

10:03 p.m. – Sacked! Sanders is taken down by Laiatu Latu, who has multiple tonight. UCLA records its sixth sack of the night with the latest.
</item 1>

<item 2>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
PENALTY – Sun Devils Ed Woods is ejected for targeting. He will miss the remainder of this game as well as the first half against Washington in two weeks.

9:10 p.m. – That's the fourth sack allowed by the Colorado O-line. The Buffaloes now lead FBS with most sacks allowed this season at 30.

8:52 p.m. – Skattebo is heating up under the sun. He's added 24 rushing yards from Arizona State's possession.

End of second quarter: Colorado 14, Arizona Sate 17

Colorado 14, Arizona State 17

8:20 p.m. FIELD GOAL – Dario Longhetto nails a 33-yard field goal to end the half. The Sun Devils have a narrow lead over the Buffs at the halfway mark.

Colorado 14, Arizona State 14

8:14 p.m. TOUCHDOWN – Sanders ties it up! The QB keeps it for 16 yards and a touchdown.
</item 2>

<item 3>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado needed a bounce back win and got one. Meanwhile Arizona State kept on struggling, falling to 1-5.

MORE: Watch Colorado vs. Arizona State live with Fubo (free trial)

The Sporting News tracked updates and highlights from college football's Week 6 game between Colorado and Arizona State.

Colorado vs. Arizona State score

Q1 Q2 Q3 Q4 Final Colorado 7 7 0 13 27 Arizona State 7 10 0 7 24

Colorado vs. Arizona State results, highlights from Week 6 game

FINAL: Colorado 27, Arizona State 24

Colorado 27, Arizona State 24

10:17 p.m. FIELD GOAL – Absolutely clutch. Mata makes a 43-yard field goal look easy to give Colorado the lead with 12 seconds on the clock.

10:14 p.m. – Bullseye! Sanders finds Antonio on a 43-yard pass. Colorado takes over on the ASU 25.
</item 3>

<item 4>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
– Colorado settles for a field goal again, this time for a 39-yarder. It leads UCLA 6-0 with 7:44 remaining in the first quarter.

7:59 p.m. INTERCEPTION – Welcome to Hollywood, Travis Hunter! The two-way star picks off Ethan Garbers at the Bruins' 36-yard line. After review, the
interception stands, and it's Buffalo ball. Colorado defense with a statement play early on.

WHAT A PICK BY TRAVIS HUNTER 🔥@CUBuffsFootball pic.twitter.com/3i9twSG1uq — FOX College Football (@CFBONFOX) October 29, 2023

Colorado 3, UCLA 0

7:54 p.m. FIELD GOAL – Sanders can't find the end zone, but Alejandro Mata finds space between the uprights. His 31-yard field goal is good, and the Buffs take
the first lead of the game on their first drive.

7:52 p.m.
</item 4>

<item 5>
Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines (Sporting News, 2023-11-25T23:20:56+00:00)
"For the big boys, for Blake, for myself, that's music to our ears that he has the confidence in us to go get the extra yard or two yards. That's who we are."

Michigan played to win the game. Ohio State, not so much. The Buckeyes settled for a 52-yard field goal late in the first half, which Jayden Fielding missed
wide left.

Even when Ohio State tied the score at 17-17 with 6:08 left in the third quarter on an up-tempo drive that featured eight straight run plays and a TreVeyon
Henderson touchdown, Michigan answered. The Wolverines scored on every drive in the second half – three field goals and a touchdown.

"It goes back to how they prepare, how they attack things and how they work," Moore said. "I would be doing them a disservice if I went out there and was
conservative."

Say what you will about the adults who were involved in the in-person scouting scandal.
</item 5>

<item 6>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado fans, you're back. For now.

A 43-yard pass from Shedeur Sanders on the Buffaloes' final drive helped set up Alejandro Mata for the game-winning 43-yard field goal with 12 seconds left,
delivering Colorado the 27-24 win over Arizona State for their first Pac-12 win of the season. It snaps their eight-game conference losing streak.

Sanders had a mediocre day by his lofty standards, but his 239 yards and a touchdown were enough in perhaps Colorado's most crucial game so far. It brought the
team to 4-2, but had it lost, it would have been much harder for the Buffs to finish the season bowl eligible. A win gives them a leg up in a competitive
conference, particularly facing four ranked teams down the stretch of the season.

If Colorado wants to solidify a bowl game, it'll have to fix many things, one of them being the offensive line.
</item 6>

<item 7>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
COURTLAND SUTTON INCREDIBLE 4TH DOWN GRAB ‼️



🎥: @NFL

pic.twitter.com/T0Mrch9Tf9 — The Sporting News (@sportingnews) November 20, 2023

Vikings 20, Broncos 15

11:02 p.m.: FIELD GOAL — Greg Joseph knocks it in from 30 yards. The Vikings' offense was moving, and it did its job by chewing a lot of clock on a 14-play,
seven-minute and 20-second drive. Minnesota stalled out on Denver's 12-yard line, settling for a field goal.

Can the Vikings' defense keep the Broncos out of the end zone as it has all night? The drive of the game is coming up.

10:57 p.m.: The spot on the field was overturned! Ty Chandler was ruled just short of the line to gain, setting up a 4th & inches.
</item 7>

<item 8>
Ole Miss vs. Georgia live score, updates, highlights from Week 11 college football game (Sporting News, 2023-11-11T22:23:02+00:00)
7. The Dawgs have showed no signs of slowing down, scoring 31 answered points against Ole Miss.

He's still HIM



Watch live on ESPN#GoDawgs pic.twitter.com/B5aa6zGpOV — Georgia Football (@GeorgiaFootball) November 12, 2023

9:48 p.m. – With his 51-yard run to help wrap up the third quarter, Kendall Milton sets a career-high in single-game rushing yards. He has 127 yards entering
the final frame.

End of third quarter: Ole Miss 14, Georgia 38

9:37 p.m. – Dart is down after combining with Zion Logue and Daylen Everette. He enters the injury tent and is replaced by Spencer Sanders.

Ole Miss 14, Georgia 38

9:32 p.m. FIELD GOAL – Peyton Woodring adds three for the Bulldogs with a 27-yard field goal, his first attempt of the night.

9:29 p.m.
</item 8>

<item 9>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado takes over on the ASU 25.

SHEDEUR TO ANTONIO FOR A HUGE GAIN 🦬@CUBuffsFootball pic.twitter.com/tvfWjRWWAE — FOX College Football (@CFBONFOX) October 8, 2023

Colorado 24, Arizona State 24

10:11 p.m. TOUCHDOWN – Don't count the Sun Devils out yet! Bourguet finds Troy Omeire on a 15-yard pass to tie it up. Only 50 seconds remain.

9:48 p.m. – It's blocked! Carter Stoutmire gets in front of the 52-yard field goal attempt from Longhetto to deny a Sun Devil score.

9:38 p.m. FIELD GOAL – Alejandro Mata nails a 42-yard field goal to extend Colorado's lead to seven. Ten minutes are left in the game.

9:35 p.m. – That's a huge run for the Buffaloes!
</item 9>

<item 10>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Vikings decided to go for it and Joshua Dobbs picked up the first down on a QB sneak.

10:54 p.m.: Ty Chandler continues to rumble. The spell RB picked up a massive gain on 3rd & 20 and it appeared that he got the first down, but the spot on the
field is under review. This could be a crucial call.

10:48 p.m.: It looked like it was going to be a three-and-out for the Vikings offense but a fake punt has kept the drive alive! Ty Chandler picked up 31 yards
on a gutsy call from head coach Kevin O'Connell to bring the ball into Denver's territory.

Vikings 17, Broncos 15

10:43 p.m.: FIELD GOAL — Will Lutz knocks one in from 28 yards, his fifth made field goal of the day. Lutz has been the Broncos' lone source of offense,
bringing Denver within one score with 10:37 to play in the game.
</item 10>
=== END ITEM id=d940616aad36e17f ===

=== ITEM id=c92f04af3e2dd68d ===
Gold sentences and facts:
Sentence 0: “The company’s approach also raises concerns regarding the GDPR,” Pachl further noted.
  Fact 0.0: The company's approach raises concerns regarding the GDPR.
  Fact 0.1: Pachl further noted this.
Sentence 1: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents.
  Fact 1.0: Meta wants to shift back to the app stores the burden of monitoring teens' social media usage.
  Fact 1.1: Meta wants to shift back to parents the burden of monitoring teens' social media usage.
Sentence 2: Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.
  Fact 2.0: Al Saadi said that her frustration with Meta is common among Palestinians.
  Fact 2.1: Al Saadi said that her frustration with Meta is common among Palestinians in occupied territories.
  Fact 2.2: Al Saadi said that her frustration with Meta is common among Palestinians across the diaspora.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.
</item 2>

<item 3>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
These included findings made by the Court of Justice of the European Union when delivering judgment on July 4, 2023 in a case in which the Court examined the
legal bases on which Meta’s processing of users’ data for behavioural advertising purposes is grounded.

“Acting in consultation with its fellow European supervisory authorities, the DPC has been engaged in a detailed regulatory assessment of the consent-based
model since it was first proposed by Meta in July. That exercise is being led by the DPC, reflecting its position as Lead Supervisory Authority for Facebook and
Instagram in Europe. The exercise has not yet concluded, and no findings have been made to date. It is due to be completed shortly, at which point the DPC will
notify Meta if it considers that the manner in which its new user offerings are to be implemented is compatible with Meta’s obligations under GDPR.”

So Meta’s move to offer users a subscription or tracking hasn’t been signed off by data protection authorities as yet — and it’s clear that further regulatory
interventions could follow.
</item 3>

<item 4>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
At the recommendation of its Oversight Board, Meta conducted a third-party due diligence report about the platform’s moderation during the May 2021 conflict.
The report found that Arabic content was flagged as potentially violating at significantly higher rates than Hebrew content was, and was more likely to be
erroneously removed. The report noted that Meta’s moderation system may not be as precise for Arabic content as it was for Hebrew content, because the latter is
a “more standardized language,” and suggested that reviewers may lack the linguistic and cultural competence to understand less common Arabic dialects like
Palestinian Arabic.

Has anything improved?

Meta committed to implementing policy changes based on the report’s recommendations, such as updating its keywords associated with dangerous organizations,
disclosing government requests to remove content and launching a hostile speech classifier for Hebrew content. Abukhater added that Meta has improved its
response to harassment, at least in comparison to other social media platforms like X (formerly Twitter).
</item 4>

<item 5>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.
</item 5>

<item 6>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Although harassment and abuse are still rampant on Instagram and Facebook, he said, the company has been responsive to suspending accounts with patterns of
targeting other users.

The company has also made more contact with regional Palestinian organizations since 2021, York added, but it’s been slow to implement recommendations from EFF
and other advocacy groups. It’s “very clear” that Meta is not putting the same resources behind Arabic and other non-English languages, York said, compared to
the attention Meta gives to countries that have the most regulatory pressure. Moderation of English and other European languages tends to be more comprehensive,
for example, because the EU enforces the Digital Services Act.

In Meta’s response to the report, Miranda Sissons, the company’s director of human rights, said that Meta was “assessing the feasibility” of reviewing Arabic
content by dialect. Sissons said that the company has “large and diverse teams” who understand “local cultural context across the region,” including in
Palestine.
</item 6>

<item 7>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
During the last violent outbreak between Hamas and Israel in 2021, a report commissioned by Meta and conducted by a third party concluded that the company’s
actions had an “adverse human rights impact” on Palestinian users’ right to freedom of expression and political participation.

Meta communications director Andy Stone declined to comment on the record, and pointed TechCrunch to Meta’s newsroom post about the conflict, which was updated
on Oct. 18 with a statement addressing the shadowbanning claims.

“Our policies are designed to keep people safe on our apps while giving everyone a voice,” the post stated. “We apply these policies equally around the world
and there is no truth to the suggestion that we are deliberately suppressing voice … We can make errors and that is why there is an appeals process for people
to tell us when they think we have made the wrong decision, so we can look into it.”

The belief that Meta shadowbans, or limits the visibility of, content about Palestine is not new.
</item 7>

<item 8>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“And so I think that is what is the most problematic right now about that particular rule, and the way that it’s enforced.”

Over the 13 days leading up to the ceasefire between Hamas and Israel, 7amleh documented more than 500 reports of Palestinian “digital rights violations,”
including the removal and restriction of content, hashtags and accounts related to the conflict.

Meta blamed some of the instances of perceived censorship to technical issues, like one that prevented users in Palestine and Colombia from posting Instagram
Stories. It attributed others to human error, like blocking the hashtag for Al-Aqsa Mosque, the holy site where Israeli police clashed with Ramadan worshippers,
because it was mistaken for a terrorist organization. The company also blocked journalists in Gaza from WhatsApp without explanation.

The same month, a group of Facebook employees filed internal complaints accusing the company of bias against Arab and Muslim users. In internal posts obtained
by BuzzFeed News, an employee attributed the bias to “years and years of implementing policies that just don’t scale globally.”
</item 8>

<item 9>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
(And on that front, Norway’s DPA, which has a local ban order on Meta’s tracking ads, told us it’s concerned about the subscription plan, saying it has doubts
as to whether it would constitute valid consent.)

As well as needing to comply with the GDPR, which delineates the qualities necessary for consent to be legal (such as that it is specific, informed and freely
given), Meta is now also subject to the pan-EU Digital Services Act (DSA) — which also sets conditions on larger platforms when it comes to tracking and
profiling people for ads. So it won’t just be data protection authorities making a call on whether Meta’s subscription or tracking offer flies; the European
Commission is responsible for oversight of very large online platforms’ DSA compliance.

Meta has also been designated as a so-called gatekeeper, under the DSA’s sister regulation, the Digital Markets Act (DMA) — which also puts some limits on use
of people’s data for ads. The Commission is the sole enforcer of the DMA.
</item 9>

<item 10>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Penalties for failing to comply with the pan-EU regulation can reach as high as 6% of global annual turnover — which, in Meta’s case, could run to multiple
billions.

The social media giant is not alone in being warned by the bloc over content concerns attached to the Israel-Hamas war: Elon Musk’s X has been singled out for
even greater attention here — with the bloc issuing both an “urgent” warning earlier this week and following that with a formal request for information about
its compliance approach.

TikTok has also received a warning from the EU about DSA content risks related to the conflict.

This report was updated with details about the operational timeline for the DSA crisis response mechanism coming into operation
</item 10>
=== END ITEM id=c92f04af3e2dd68d ===

=== ITEM id=1d862a789ee46b05 ===
Gold sentences and facts:
Sentence 0: In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds the performance of other leading gen AI models like OpenAI’s GPT-4.
  Fact 0.0: Google touted Gemini's superior architecture in blog posts and press materials.
  Fact 0.1: Google touted Gemini's superior capabilities in blog posts and press materials.
  Fact 0.2: Google claimed that Gemini meets the performance of other leading gen AI models, such as OpenAI's GPT-4.
  Fact 0.3: Google claimed that Gemini exceeds the performance of other leading gen AI models, such as OpenAI's GPT-4.
Sentence 1: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 1.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 1.1: The case argues that Google siphons off news publishers' content.
  Fact 1.2: The case argues that Google siphons off news publishers' readers.
  Fact 1.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 1.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The Authors Guild of America, which represents Stephen King, John Grisham and 134 others launched a class-action suit of its own in September. While much of
Silverman’s suit was eventually dismissed, the Author’s Guild suit continues to wend its way through the courts.

Select news outlets, on the other hand, proved far more amenable. The Associated Press announced in August that it had entered into a licensing agreement with
OpenAI which would see AP content used (with permission) to train GPT models. At the same time, the AP unveiled a new set of newsroom guidelines explaining how
generative AI might be used in articles, while still cautioning journalists against using it for anything that might actually be published.

ChatGPT itself didn’t seem too inclined to follow the rules. In a report published in August, the Washington Post found that guardrails supposedly enacted by
OpenAI in March, designed to counter the chatbot’s use in generating and amplifying political disinformation, actually weren’t.
</item 1>

<item 2>
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
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 5>

<item 6>
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
</item 6>

<item 7>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 7>

<item 8>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
This week, Google took the wraps off of Gemini, its new flagship generative AI model meant to power a range of products and services including Bard, Google’s
ChatGPT competitor. In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds
the performance of other leading gen AI models like OpenAI’s GPT-4.

But the anecdotal evidence suggests otherwise.

A “lite” version of Gemini, Gemini Pro, began rolling out to Bard yesterday, and it didn’t take long before users began voicing their frustrations with it on X
(formerly Twitter).

The model fails to get basic facts right, like 2023 Oscar winners:

I'm extremely disappointed with Gemini Pro on Bard. It still give very, very bad results to questions that shouldn't be hard anymore with RAG. A simple question
like this with a simple answer like this, and it still got it WRONG.
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
Google launched Gemini Pro in Bard, touting strong capabilities, but early user and TechCrunch tests found frequent errors: incorrect Oscar facts, fabricated
claims, failed French word-length prompts, stale or evasive news summaries, and weak coding performance. Reports also showed jailbreaks eliciting harmful
suggestions. Gemini Pro is not Google’s forthcoming Ultra model, and Google compared it with GPT-3.5; despite promised gains over Bard’s prior model, the
initial results suggest substantial room for improvement.
</item 10>
=== END ITEM id=1d862a789ee46b05 ===

=== ITEM id=f5af96ea83bfbc90 ===
Gold sentences and facts:
Sentence 0: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 0.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 0.1: Prabhakar Raghavan revealed in court testimony that Google paid $26.3 billion in 2021.
  Fact 0.2: The $26.3 billion was paid to multiple browsers, phones and platforms.
  Fact 0.3: The payments were made to companies including Apple, Samsung and Mozilla.
  Fact 0.4: The Verge reports this.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: The case (the sentence's subject, referred to as 'It') will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 3>

<item 4>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 4>

<item 5>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
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
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 7>

<item 8>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
He hoped to make Google and Apple’s services so connected as to be inseparable.

The way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention all that juicy data —
and Apple would get more revenue, as a result of the lucrative search agreement the two signed.

For whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.

Google’s deleted chat logs

During Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In February, the
DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually changed the
setting.

Pichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according to CNBC.
</item 8>

<item 9>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 10>
=== END ITEM id=f5af96ea83bfbc90 ===

=== ITEM id=b625f68637ed5437 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury.
  Fact 0.1: Rehn declared to the jury that all of that, all of it, was built on lies.
Sentence 1: Binance has made headlines this past year for a range of reasons, including Zhao’s comments contributing to the collapse of FTX, which was once one of its top competitors.
  Fact 1.0: Binance has made headlines this past year for a range of reasons.
  Fact 1.1: Binance has made headlines this past year including because of Zhao's comments.
  Fact 1.2: Zhao's comments contributed to the collapse of FTX.
  Fact 1.3: FTX was once one of Binance's top competitors.

Retrieved items:
<item 1>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 1>

<item 2>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 4>

<item 5>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”
</item 5>

<item 6>
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
</item 6>

<item 7>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
regulators and government officials, many of whom he donated to.

Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).

But in early November 2022, that all changed.

FTX’s collapse

Concerns surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8
billion in liabilities as of June 30, 2022.

But there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token
behind FTX.

The balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at
the time.
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
"FTX defrauded all its customers," he said.

Matt Huang, co-founder of Paradigm Operations LP, right, arrives at court in New York, US, on Thursday, Oct. 5, 2023. Former FTX Co-Founder Sam Bankman-Fried is
charged with seven counts of fraud and money laundering following the collapse of his cryptocurrency empire last year. Photographer: Yuki Iwamura/Bloomberg via
Getty Images Yuki Iwamura | Bloomberg | Getty Images

Investment to zero

The third witness to take the stand was Matt Huang, co-founder and managing partner of Paradigm, a crypto venture capital firm that invested over $275 million
in FTX. That stake was wiped out. Huang testified about his firm's due diligence on FTX, and he told the court that Bankman-Fried assured him that funds would
be used for FTX and not Alameda.
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 10>
=== END ITEM id=b625f68637ed5437 ===
