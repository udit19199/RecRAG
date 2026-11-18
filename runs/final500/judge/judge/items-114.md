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

=== ITEM id=fe02679c3c91df1c ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: The friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: Making the friendship bracelet seemingly expressed his interest in Swift.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: He later invited the 'Anti-Hero' singer on his podcast to one of his games at Arrowhead Stadium.
  Fact 1.1: The 'Anti-Hero' singer famously accepted the invitation on 24 September.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide.
  Fact 2.1: Swift has shared that she has nothing to hide in her relationship with Kelce.
  Fact 2.2: Swift's relationship with Kelce is new.

Retrieved items:
<item 1>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But before he could even answer, the answer was given for him.

Sept. 14: Rumors aren't denied

"The rumors are terrible and cruel, but honey, most of them are true." - New Romantics

Of course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not confirm them, but he
also did not deny them.

Sept. 17: The references begin

"I watched it begin again." - Begin Again

The rumors began to pick up, and now there were a lot of people, commentators included, making Swift references around Kelce.

When the tight end caught a touchdown pass from Patrick Mahomes, CBS Sports' Ian Eagle said, "Kelce finds a blank space for the score."

This was only the start to many, many, Swift lyrics being referenced during NFL games.

Sept. 21: Kelce sends Swift an invite

"So inviting, I almost jump in."
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.
</item 4>

<item 5>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 9>

<item 10>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 10>
=== END ITEM id=fe02679c3c91df1c ===

=== ITEM id=a378b09300492f82 ===
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
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 1>

<item 2>
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
</item 2>

<item 3>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
It replied with an automated email reading: “Busy now, please check back later.”

Update: X’s global government affairs account has posted a thread responding to the EU’s criticism in which it claims the data shows other social networking
services saw greater changes in subscriber growth and total growth for pro-Kremlin accounts, and for engagement and volume; it also writes that: “More than 700
unique Community Notes have appeared on posts related to the Ukraine conflict. Through Community Notes’ new ‘Notes on Media’ feature, which automatically
matches notes on an image or video to other posts that contain matching media, these notes appeared on over 1,200 additional posts. These notes are in numerous
languages and are written specifically for local audiences.”

Back in April, X/Twitter was named as one of 19 larger platforms that face the strictest level of regulation under the EU’s DSA.
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
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 5>

<item 6>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 6>

<item 7>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
X (formerly Twitter) has been called out in the European Union for having the worst ratio of disinformation/misinformation to posts not spreading falsehoods
among mainstream social networks.

In a speech today discussing the latest updates from other platforms that have signed up to combat disinformation under a beefed up EU Code of Practice that’s
been linked to compliance with the the bloc’s legally binding Digital Services Act (DSA), it commissioner for values and transparency, Vera Jourova, said X’s
poor performance was assessed during a pilot phase of testing of new methodology developed by Code signatories.

“A big novelty under the Code is now the publication of the first set of structural indicators such as how easy it is to find disinformation content, how much
engagement such content receives or indicators about sources. These indicators — developed by the signatories — constitute an unprecedented and novel insight
into disinformation on online platforms,” she suggested. “Such insight is crucial to understand how effective platforms efforts are to fight this threat even
more efficiently.
</item 7>

<item 8>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
The law puts an obligation on so-called very large online platforms (VLOPs) to assess societal risks attached to their use of algorithms — and put in place
“reasonable, proportionate and effective mitigation measures” for identified risks, with their reporting and mitigation plans subject to independent audit and
oversight by the Commission.

We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.

Update: A Commission spokesperson told us:

While the Code is voluntary, compliance with the DSA is not. The DSA is now legally binding on X/Twitter. This includes in particular the obligation of Very
Large Online Platforms, such as X/Twitter to mitigate the risks their platforms pose for the spread of disinformation. The TrustLab study to measure two
Structural Indicators covers signatories of the Code at the time of the launch of the exercise, which at the time included X/Twitter.
</item 8>

<item 9>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
On Wednesday, in an “urgent” letter to Musk, the EU’s internal market commissioner Thierry Breton said the bloc had seen “indications” from “qualified sources”
that X is being used to disseminate illegal content and disinformation in the EU” following Saturday’s attacks — before reminding X of the DSA’s “very precise
obligations” vis-à-vis content moderation.

“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” Breton also warned then.

In its latest press release, the EU said it has sent X a formal request for information under the DSA.

“This request follows indications received by the Commission services of the alleged spreading of illegal content and disinformation, in particular the
spreading of terrorist and violent content and hate speech,” it wrote, adding that the ask also addresses other aspects of DSA compliance.

The pan-EU regulation puts a series of governance obligations on digital services and platforms with the aim of ensuring companies are responding to reports of
illegal content.
</item 9>

<item 10>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In this case, failure to reply by the deadline could lead to the imposition of period penalties,” the EU added.

Earlier this year, Musk pulled out of the EU’s Code of Practice on online disinformation. In a response to X leaving the voluntary Code, Breton warned:
“Obligations remain. You can run but you can’t hide.”

In recent days, TikTok and Meta have also been warned by the EU over disinformation related to the Israel-Hamas war. But back in September, a study commissioned
via the EU’s Code of Practice suggested X is the worst of the major platforms when it comes to spreading disinformation.

This report was substantially revised to clarify a number of aspects, including — most saliently — that the EU has sent a formal request for information to X at
this point. This could prefigure a formal announcement of an investigation but is not, yet, that formal technical step.
</item 10>
=== END ITEM id=a378b09300492f82 ===

=== ITEM id=9068ed8b4949d159 ===
Gold sentences and facts:
Sentence 0: “There was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic,” said an Anthropic spokesperson, Sally Aldous.
  Fact 0.0: An Anthropic spokesperson, Sally Aldous, said there was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic.
  Fact 0.1: Sally Aldous is an Anthropic spokesperson.
Sentence 1: Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.
  Fact 1.0: Arun Chandrasekaran is a Gartner analyst.
  Fact 1.1: Arun Chandrasekaran said Altman's exit is indeed shocking.
  Fact 1.2: Arun Chandrasekaran said Altman has been the face of generative AI technology.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory right now is based on the board's language.
  Fact 2.1: The prevailing theory right now is based on the way giant tech companies work.
  Fact 2.2: The prevailing theory is floating around right now.

Retrieved items:
<item 1>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 1>

<item 2>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He said OpenAI still has a “deep bench of technical leaders” but its next executives will have to steer it through the challenges of scaling the business and
meeting the expectations of regulators and society.

Forrester analyst Rowan Curran speculated that Altman’s departure, “while sudden,” did not likely reflect deeper business problems.

“This seems to be a case of an executive transition that was about issues with the individual in question, and not with the underlying technology or business,”
Curran said.

Altman has a number of possible next steps. Even while running OpenAI, he placed large bets on several other ambitious projects.

Among them are Helion Energy, for developing fusion reactors that could produce prodigious amounts of energy from the hydrogen in seawater, and Retro
Biosciences, which aims to add 10 years to the human lifespan using biotechnology.
</item 2>

<item 3>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“The co-founders themselves came to the conclusion that they wished to depart OpenAI to start their own company, made this known to OpenAI’s leadership, and
over several weeks negotiated an exit on mutually agreeable terms.” Anthropic accepted a $US4 billion investment from Amazon and another $US2 billion from
Google two years later. The Reveal After OpenAI received another $US2 billion from Microsoft, Altman and another senior executive, Greg Brockman, visited Bill
Gates at his sprawling mansion on the shores of Lake Washington, outside Seattle. The Microsoft founder was no longer involved in the company day to day but
kept in regular touch with its executives.

Over dinner, Gates told them he doubted that large language models could work. He would stay sceptical, he said, until the technology performed a task that
required critical thinking – passing an Advanced Placement biology test, for instance. Five months later, on August 24, 2022, Altman and Brockman returned and
brought along an OpenAI researcher named Chelsea Voss.
</item 3>

<item 4>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Altman and others pushed back, Musk quit and said he would focus on his own AI work at Tesla. In February 2018, he announced his departure to OpenAI’s
staff on the top floor of the startup’s offices in a converted truck factory, three people who attended the meeting said. When he said that OpenAI needed to
move faster, one researcher retorted at the meeting that Musk was being reckless. Musk called the researcher a “jackass” and stormed out, taking his deep
pockets with him. OpenAI suddenly needed new financing in a hurry. Altman flew to Sun Valley for a conference and ran into Satya Nadella, Microsoft’s CEO. A
tie-up seemed natural. Altman knew Microsoft’s chief technology officer, Kevin Scott. Microsoft had bought LinkedIn from Hoffman, an OpenAI board member.
Nadella told Scott to get it done. The deal closed in 2019.
</item 4>

<item 5>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He’s sat with multiple heads of state to discuss AI’s potential and perils. Just Thursday, he took part in a CEO summit at the Asia-Pacific Economic Cooperation
conference in San Francisco, where OpenAI is based.

He predicted AI will prove to be “the greatest leap forward of any of the big technological revolutions we’ve had so far.” He also acknowledged the need for
guardrails, calling attention to the existential dangers future AI could pose.

Some computer scientists have criticized that focus on far-off risks as distracting from the real-world limitations and harms of current AI products. The U.S.
Federal Trade Commission has launched an investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false
information through its chatbot.
</item 5>

<item 6>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Thank you… — Eric Schmidt (@ericschmidt) November 17, 2023

OpenAI’s board of directors now consists of OpenAI’s chief scientist Ilya Sutskever; Quora CEO Adam D’Angelo; tech entrepreneur Tasha McCauley; and Helen Toner,
the director of strategy at Georgetown University’s Center for Security and Emerging Technology.

Altman’s removal comes unexpectedly, to say the least. The company just last week hosted its first developer conference, OpenAI DevDay, which Altman emceed.
Altman spoke at the Asia-Pacific Economic Cooperation conference and an Oakland, California event on Thursday. And according to The Verge and The New York
Times, citing multiple internal sources, OpenAI employees learned of Altman’s firing when it was announced publicly.

Altman has a long history steering OpenAI.
</item 6>

<item 7>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
</item 7>

<item 8>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
The company said its board consists of OpenAI’s chief scientist, Ilya Sutskever, and three non-employees: Quora CEO Adam D’Angelo, tech entrepreneur Tasha
McCauley, and Helen Toner of the Georgetown Center for Security and Emerging Technology.

OpenAI’s key business partner, Microsoft, which has invested billions of dollars into the startup and helped provide the computing power to run its AI systems,
said that the transition won’t affect its relationship.

“We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of AI to our customers,” said an
emailed Microsoft statement.

While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s. He was recruited in 2014 to take lead of
the startup incubator YCombinator.
</item 8>

<item 9>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
ChatGPT-maker Open AI said Friday it has pushed out its co-founder and CEO Sam Altman after a review found he was “not consistently candid in his
communications” with the board of directors.

“The board no longer has confidence in his ability to continue leading OpenAI,” the artificial intelligence company said in a statement.

In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial
intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.

Mira Murati, OpenAI’s chief technology officer, will take over as interim CEO effective immediately, the company said, while it searches for a permanent
replacement.

The announcement also said another OpenAI co-founder and top executive, Greg Brockman, the board’s chairman, would step down from that role but remain at the
company, where he serves as president.
</item 9>

<item 10>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Around the same time, Musk, who had co-chaired its board with Altman, resigned from the board in a move that OpenAI said would eliminate a “potential future
conflict for Elon” due to Tesla’s work on building self-driving systems.

While OpenAI’s board has preserved its nonprofit governance structure, the startup it oversees has increasingly sought to capitalize on its technology by
tailoring its popular chatbot to business customers.

At its first developer conference last week, Altman was the main speaker showcasing a vision for a future of AI agents that could help people with a variety of
tasks. Days later, he announced the company would have to pause new subscriptions to its premium version of ChatGPT because it had exceeded capacity.

Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.
</item 10>
=== END ITEM id=9068ed8b4949d159 ===

=== ITEM id=7306270cd32c313e ===
Gold sentences and facts:
Sentence 0: The current tour of India is not a multi-format series, but the defeat in the one-off Test marks Australia’s first loss in a four-day game since January 2014.
  Fact 0.0: The current tour of India is not a multi-format series.
  Fact 0.1: The defeat in the one-off Test was Australia's first loss in a four-day game since January 2014.
Sentence 1: This was one of the closest matches that have been played between these two sides, as Australia won the match by just one run.
  Fact 1.0: The match was one of the closest matches played between these two sides.
  Fact 1.1: Australia won the match by just one run.
Sentence 2: Australian Man-of-the-Match Steve Smith (105 runs in 93 balls) and Aaron Finch (81 runs in 116 balls) were the guiding forces in taking Australia to 328 runs for 7 wickets in 50 overs.
  Fact 2.0: Steve Smith was Australia's Man-of-the-Match.
  Fact 2.1: Steve Smith scored 105 runs in 93 balls.
  Fact 2.2: Aaron Finch scored 81 runs in 116 balls.
  Fact 2.3: Steve Smith and Aaron Finch were the guiding forces in taking Australia to 328 runs for 7 wickets in 50 overs.

Retrieved items:
<item 1>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Alyssa Healy has rejected any suggestion Australia’s stranglehold on international women’s cricket is slipping, saying their Test loss to India is barely a blip
on the team’s record.

Australia stumbled to their first Test defeat to India by eight wickets in Mumbai on Sunday, after being outplayed by the hosts in foreign spinning conditions.

The loss comes after one of the more difficult periods for the Australian side in the past six years, with the retirements of Meg Lanning and Rachael Haynes.

Facebook Twitter Whatsapp Reddit Email Share

After winning their fourth straight global ICC event earlier this year in the 50-over World Cup, Australia retained the Ashes via a drawn multi-format series in
England in July.

That tour included losing four of six white-ball matches, with Australia’s win in the Test enough to have them level on points with England and hold onto the
trophy.
</item 1>

<item 2>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India moved to finals and beat Sri Lanka to win their second World Cup title.



2015 World Cup, against Australia

India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26. Australia outplayed India in almost all departments
of the game.

Australian Man-of-the-Match Steve Smith (105 runs in 93 balls) and Aaron Finch (81 runs in 116 balls) were the guiding forces in taking Australia to 328 runs
for 7 wickets in 50 overs.

In reply, India needed a few big innings, but apart from skipper MS Dhoni (65 runs in 65 balls), Shikhar Dhawan (45 runs in 34 balls), and Ajinkya Rahane (44
runs in 68 balls), the batting order collapsed miserably as India lost the match by 95 runs.



2019 World Cup, against New Zealand

The cricket World Cup returned to England for the fifth time.
</item 2>

<item 3>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Damien Fleming took five wickets for Australia as Australia won the match by 16 runs.

1999, 1st Super, ICC World Cup, The Oval

The Australian dominance continued in this World Cup as well, as they defeated India by 77 runs. Australia posted a target of 283 runs in 50 overs courtesy of a
brilliant 83 of 99 balls by Mark Waugh.

India had an early batting collapse but was stabilised by a sensible 100 off 138 balls by Ajay Jadeja. Robin Singh also scored 75 runs off 94 balls. However,
India was bowled out within 205 runs in 48.2 overs. Glenn McGrath picked up three wickets for Australia, with Damien Fleming and Steve Waugh picking up two
each.

2003, Match 11, ICC World Cup, Centurion

This was India's biggest batting collapse in the history of the ODI World Cup.
</item 3>

<item 4>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India, one of the favourites to win the World Cup, were very much on the path to victory till Mohammad Azharuddin, who top-scored with 64 runs was at the
crease.

But with the fall of his wicket, the Indian lower order collapsed like a house of placards as India lost the match by 35 runs.

1996 Wills World Cup, against Sri Lanka

The 1996 Cricket World Cup is known for Sri Lanka's emergence as a cricketing power on the horizon of world cricket and the beginning of Sachin Tendulkar's
extraordinary dominance in batting history.

Playing in the first semi-final at the Eden Gardens in Kolkata on March 13, India were favourites again, and they raised their hopes when they managed to
restrict Sri Lanka to 251 for 8 in 50 overs despite Man-of-the-Match Aravinda de Silva's 66 and Roshan Mahanama's 58.
</item 4>

<item 5>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
This innings, by Ponting, had famously created the conspiracy that he had Spring in his bat.

India gave up a tough fight, with Virender Sehwag making 82 runs off 81 balls and Rahul Dravid also scoring 47 runs off 57 balls. However, the lack of support
from the other batters led to India being bowled out within 234 runs. Brett Lee and Andrew Symonds took two wickets each as Australia lifted the World Cup for
the third time.

Also Read: India vs Australia, World Cup final in 2003 vs 2023: Can India change the course of history in the final?

2011, 2nd Quarter-Final, ICC Cricket World Cup, Ahmedabad

After so much dominance by the Australian side in the World Cups against India, India finally got their revenge in this World Cup as they defeated them by five
wickets.
</item 5>

<item 6>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
In reply, Mohinder Amarnath played a useful innings of 46, while Yashpal Sharma scored a fine 61.

Sandeep Patil played a fiery innings of 51 not out in just 32 balls as India stormed into semifinals for the first time, beating England by six wickets with 5.2
overs to spare.

Also Read: ICC World Cup 2023: List of centuries; Most tons in World Cup so far

1987 Reliance Cup, against England

Four years later, the World Cup was being played outside England for the first time, and it was India which was hosting it.

In a co-incidence, India was against England in the semi-finals once again, at the Wankhede Stadium, Mumbai, on November 5.

This time, England's Man-of-the-Match skipper Graham Gooch smashed 115 runs in 136 balls to lay a solid foundation for his team's fighting total of 254 for 6
wickets.
</item 6>

<item 7>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
The current tour of India is not a multi-format series, but the defeat in the one-off Test marks Australia’s first loss in a four-day game since January 2014.

But quizzed over whether the defeat suggested Australia’s unprecedented period of dominance was ending, Healy was defiant.

“It’s probably not even a blip,” Healy said.

Advertisement

“It (Test cricket) is a format where we’re not overly familiar with, in conditions that we’re not familiar with.

“I see more positives out of this and than any sort of negativity.

“There is always a lot of expectation and external noise about our team and how we’re performing.

“They are so used to seeing us winning, that I think, at times, they forget that we’ve actually seen quite a lot of change over the last 12 to 18 months.

“That kind of goes unnoticed external to our group.”

Australia have spoken about shifting their mindset in white-ball cricket after the losses in England, pursuing a more attacking approach.
</item 7>

<item 8>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia posted a target of 329 runs, losing seven wickets in the process. Steven Smith scored a brilliant 105 runs off 93 balls, with Aaron Finch supporting
the innings with an 81-run knock of 116 balls. Umesh Yadav picked up four wickets for India. India crumbled within 233 runs in 46.5 overs. MS Dhoni was the
highest scorer for the Indian side in that match, with 65 runs off 65 balls.

James Faulkner picked up three wickets for Australia, with Mitchell Johnson and Mitchell Stark taking two wickets each.

Australia went all the way to lift their fifth ODI World Cup trophy, defeating New Zealand in the final.

2019, Match 14, ICC Cricket World Cup, The Oval

India took their revenge in this World Cup as they annihilated the Australian bowlers by posting a target of 353 runs in 50 overs.
</item 8>

<item 9>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia batted first and posted a target of 261 runs courtesy of a 104-run knock of 118 balls by their captain, Ricky Ponting, with the loss of six wickets in
the process.

In the second innings, Sachin Tendulkar, Yuvraj Singh, and Gautam Gambhir scored three half-centuries as India changed down the target within 47.4 overs, losing
5 wickets in the process.

India went all the way to lift their second ODI World Cup trophy, defeating Sri Lanka in the finals.

2015, 2nd Semi-Final, ICC Cricket World Cup, Sydney

Australia got their revenge for the 2011 quarterfinals in this World Cup as they defeated India by 95 runs. Australia posted a target of 329 runs, losing seven
wickets in the process.
</item 9>

<item 10>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India seamer Javagal Srinath took three top-order wickets, including Sanath Jayasuriya and Romesh Kaluwitharana.

Fans at the Eden Garden were cheering loudly till Tendulkar was at the crease.

But once he fell for 65 with the team at 98, India suffered one of the worst collapses in the batting history of the World Cups.

The crowd turned angry and hostile towards Sri Lanka, creating a ruckus at the stadium, and the match was interrupted.

With no respite from onlookers' side, umpires decided to call off the match when India were reeling at 120 for eight in 34.1 overs.

The long-lasting memories of the crowd's bad behaviour and a weeping Vinod Kambli returning to the pavilion are still fresh in the memories of the cricket fans
of that generation.
</item 10>
=== END ITEM id=7306270cd32c313e ===

=== ITEM id=3caa00c4e2b33959 ===
Gold sentences and facts:
Sentence 0: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 0.0: The planned argument will attempt to argue that Google restrains competition within two separate markets.
  Fact 0.1: The two markets include the distribution of apps to Android users.
  Fact 0.2: The two markets include the market for payment processing solutions for content inside Android apps.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed the article for not meeting its standards.
  Fact 1.1: The degree to which the article was shared on social media shows just how willing people are to accept foul play on Google's part.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The degree to which the article was boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google's part.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 2>

<item 3>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 4>

<item 5>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 5>

<item 6>
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
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 7>

<item 8>
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
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 10>
=== END ITEM id=3caa00c4e2b33959 ===
