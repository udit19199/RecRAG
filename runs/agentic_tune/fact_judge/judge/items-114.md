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

=== ITEM id=28e4d94ab1a4ac9f ===
Gold sentences and facts:
Sentence 0: Because GPT-4 is more likely to follow the instructions of “jailbreaking” prompts, the co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
  Fact 0.0: GPT-4 is more likely to follow the instructions of 'jailbreaking' prompts.
  Fact 0.1: The co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
  Fact 0.2: The co-authors claim GPT-4 can be more easily prompted to spout toxic, biased text because it is more likely to follow jailbreaking prompts.
Sentence 1: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 1.0: The entity's earlier actions were focused on concerns about the spread of illegal content related to the Israel-Hamas war.
  Fact 1.1: The entity's earlier actions were focused on concerns about the spread of disinformation related to the Israel-Hamas war.

Retrieved items:
<item 1>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
But at the time of writing, Musk had engaged in a bit of a dance of his own with the news of the EU’s warning by responding to a critical tweet posted to X by
journalist Glenn Greenwald, who attacked the EU’s new “censorship law,” as he dubbed the DSA — which he claimed would be used to “punish X.”

In reply to Greenwald, Musk avoided expressing the same trenchant criticism of the EU’s approach but invited yet more relativism — writing: “Let the public hear
exactly what this disinformation consists of and decide for themselves.”

He then went on to sew doubt that disinformation is something that can be independently arbitrated at all, implying fact-checking is just a convenient exercise
for targeting different opinions — so essentially echoing Greenwald’s position — by claiming that “many times” the “official fact-checker” has been found making
false statements, before adding a rhetorically empty offer that “Maybe this is the case here, maybe not” for good, plausible deniability measure.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> disinformation
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> disinformation

Entities: disinformation, EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks, illegal content, Glenn
Greenwald, fact-checking, EU, TechCrunch, Hamas attacks, X, Elon Musk, Hamas, DSA
</item 1>

<item 2>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> hate speech
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Saturday’s attacks
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> terrorist and violent content
- Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design -MENTIONS-> risk assessment
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Commission services

Entities: Investigation into X over alleged disinformation, complaint handling, risk assessment, After warning, the EU opens investigation into X over alleged
disinformation, disinformation, illegal content, EU, TechCrunch, X, Elon Musk, Commission, DSA, Twitter
</item 2>

<item 3>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> illegal content
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> illegal content
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> illegal content
- Meta says it’s prioritizing livestreaming checks during Israel-Hamas war -MENTIONS-> illegal content
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> illegal content

Entities: illegal content, fact-checking, moderation, TechCrunch, EU investigation of X, deceptive UX design, info manipulation, Elon Musk’s X faces first DSA
probe in EU over illegal content risks, moderation, transparency and deceptive UX design, X, EU, Elon Musk, transparency, Community Notes, Commission, DSA
</item 3>

<item 4>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> disinformation
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> disinformation

Entities: disinformation, Musk, EU, TechCrunch, X, propaganda, Community Notes, X (formerly Twitter) is worst for disinformation, per EU analysis
</item 4>

<item 5>
Graph query row:
article: Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design
publisher: TechCrunch
passages: ['Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch,
2023-12-18T13:04:56+00:00)\nIt’s official! Elon Musk’s X marks the spot of the first confirmed investigation opened by the European Union under its rebooted
digital rulebook, the Digital Services Act (DSA).\n\nAnnouncing the opening of a “formal proceeding” today, the European Commission said the investigation will
look at whether the social networking platform may have breached rules linked to risk management, content moderation, dark patterns, advertising transparency
and data access for researchers.\n\nThe opening of a formal DSA investigation on X comes hard on the heels of a complaint against X’s adtech by privacy rights
group, noyb — although today’s formal proceeding is unlikely to be directly linked as the Commission has been probing the platform for months via a flurry of
requests for information. Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas
war.\n\nBack in October the Commission sent Musk’s company an “urgent” formal request for information about how it was responding to information risks arising
out of the conflict.', 'Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch,
2023-12-18T13:04:56+00:00)\nThe EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices
on illegal content, complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to
respond.\n\n“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X
in September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.', 'Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency
and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)\nPer the Commission, the EU’s investigation of X focuses on the following areas and
issues:\n\nCompliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment
and mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism
for illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources\n\n— “ The effectiveness of measures taken by X to
combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing fact-checking to aggregated views of users, who
may vote to label disputed tweets with additional context.', 'Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and
deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)\nThe EU said the probe will also look at the effectiveness of “related policies mitigating risks to
civic discourse and electoral processes”\n\n— with a focus on the effectiveness of its “Community Notes” system of crowdsourcing fact-checking to aggregated
views of users, who may vote to label disputed tweets with additional context. The EU said the probe will also look at the Transparency requirements — with a
focus on “suspected shortcomings” by X in giving researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected shortcomings
in its ads repository (aka the ads transparency library the regulation also mandates)\n\n— with a focus on “suspected shortcomings” by X in giving researchers
access to publicly accessible data as mandated by Article 40 of the DSA, and suspected shortcomings in its ads repository (aka the ads transparency library the
regulation also mandates) Suspected deceptive design elements in X’s user interface — “notably in relation to checkmarks linked to certain subscription
products, the so-called Blue checks”\n\n“If proven,', 'Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and
deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)\nthe so-called Blue checks”\n\n“If proven, these failures would constitute infringements of Articles
34(1), 34(2) and 35(1), 16(5) and 16(6), 25(1), 39 and 40(12) of the DSA,” the EU added — saying it will now carry out an in-depth investigation “as a matter of
priority”.\n\nToday we open formal infringement proceedings against @X : ⚠️ Suspected breach of obligations to counter #IllegalContent and #Disinformation ⚠️
Suspected breach of #Transparency obligations ⚠️ Suspected #DeceptiveDesign of user interface#DSA pic.twitter.com/NxKIif603k — Thierry Breton (@ThierryBreton)
December 18, 2023\n\nThe Commission is responsible for enforcing the DSA on larger platforms such as X — which was designated a very large online platform
(VLOP) under the DSA back in April.', 'Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design
(TechCrunch, 2023-12-18T13:04:56+00:00)\nReached for a response to the EU probe, X’s Joe Benarroch — who was brought into the company by CEO Linda Yaccarino
relatively recently to work on business operations — said:\n\nX remains committed to complying with the Digital Services Act, and is cooperating with the
regulatory process. It is important that this process remains free of political influence and follows the law. X is focused on creating a safe and inclusive
environment for all users on our platform, while protecting freedom of expression, and we will continue to work tirelessly towards this goal.\n\nWhile X under
Musk has very evidently been flying the ship in the opposite direction to the responsible governance the DSA intends to encourage — hence attracting so many
warnings from EU lawmakers, even long before today’s formal probe — it’s notable X recently started offering a research program for outsiders to get data to
systemic risks arising on the platform in the EU. However the Commission obviously has doubts X has gone far enough on the transparency front to meet the DSA’s
bar.', 'Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch,
2023-12-18T13:04:56+00:00)\nEU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced
opinions is a valid approach to tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and
processes has cost X’s users in terms of harm exposure.\n\nTo say the Commission’s investigation looks fascinating and timely is an understatement.\n\nEU
lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.\n\nThe investigation may also test Musk’s mettle for what could be an expensive
head-on clash with EU regulators.', 'Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design
(TechCrunch, 2023-12-18T13:04:56+00:00)\nTime will tell how this one plays out but he may find the dead blue bird he’s chosen to hang around his neck, when he
purposefully ripped up the standard content moderation rulebook and opted to channel online chaos, starting to feel like it’s dragging him down.']
connections: [{'related': 'risk assessment', 'relationship': 'MENTIONS'}, {'related': 'systemic risks', 'relationship': 'MENTIONS'}, {'related': 'Linda
Yaccarino', 'relationship': 'MENTIONS'}, {'related': 'disinformation', 'relationship': 'MENTIONS'}, {'related': 'illegal content', 'relationship': 'MENTIONS'},
{'related': 'fact-checking', 'relationship': 'MENTIONS'}, {'related': 'moderation', 'relationship': 'MENTIONS'}, {'related': 'EU', 'relationship': 'MENTIONS'},
{'related': 'Freedom of expression', 'relationship': 'MENTIONS'}, {'related': 'risk management', 'relationship': 'MENTIONS'}, {'related': 'content moderation',
'relationship': 'MENTIONS'}, {'related': 'European Commission', 'relationship': 'MENTIONS'}, {'related': 'formal DSA investigation on X', 'relationship':
'MENTIONS'}, {'related': 'noyb', 'relationship': 'MENTIONS'}, {'related': 'X’s adtech', 'relationship': 'MENTIONS'}, {'related': 'complaint against X’s adtech',
'relationship': 'MENTIONS'}, {'related': 'dark patterns', 'relationship': 'MENTIONS'}, {'related': 'advertising transparency', 'relationship': 'MENTIONS'},
{'related': 'data access for researchers', 'relationship': 'MENTIONS'}, {'related': 'formal request for information', 'relationship': 'MENTIONS'}, {'related':
'formal infringement proceedings', 'relationship': 'MENTIONS'}, {'related': 'Hamas’ terrorist attacks against Israel', 'relationship': 'MENTIONS'}, {'related':
'EU investigation of X', 'relationship': 'MENTIONS'}, {'related': 'deceptive UX design', 'relationship': 'MENTIONS'}, {'related': 'info manipulation',
'relationship': 'MENTIONS'}, {'related': 'first DSA probe in EU', 'relationship': 'MENTIONS'}, {'related': 'illegal content risks', 'relationship': 'MENTIONS'},
{'related': 'civic discourse', 'relationship': 'MENTIONS'}, {'related': 'electoral processes', 'relationship': 'MENTIONS'}, {'related': 'Article 40 of the DSA',
'relationship': 'MENTIONS'}, {'related': 'ads repository', 'relationship': 'MENTIONS'}, {'related': 'Blue checks', 'relationship': 'MENTIONS'}, {'related':
'user interface', 'relationship': 'MENTIONS'}, {'related': 'subscription products', 'relationship': 'MENTIONS'}, {'related': 'X', 'relationship': 'MENTIONS'},
{'related': 'formal infringement proceedings against X', 'relationship': 'MENTIONS'}, {'related': 'deceptive design of user interface', 'relationship':
'MENTIONS'}, {'related': 'infringements of DSA provisions', 'relationship': 'MENTIONS'}, {'related': 'Joe Benarroch', 'relationship': 'MENTIONS'}, {'related':
'EU probe', 'relationship': 'MENTIONS'}, {'related': 'research program', 'relationship': 'MENTIONS'}, {'related': 'EU regulators', 'relationship': 'MENTIONS'},
{'related': 'the Commission', 'relationship': 'MENTIONS'}, {'related': 'crowdsourced opinions', 'relationship': 'MENTIONS'}, {'related': 'diverse information
risks', 'relationship': 'MENTIONS'}, {'related': 'harm exposure', 'relationship': 'MENTIONS'}, {'related': 'first DSA probe', 'relationship': 'MENTIONS'},
{'related': 'Elon Musk', 'relationship': 'MENTIONS'}, {'related': 'Israel-Hamas war', 'relationship': 'MENTIONS'}, {'related': 'European Union', 'relationship':
'MENTIONS'}, {'related': 'Hamas', 'relationship': 'MENTIONS'}, {'related': 'transparency', 'relationship': 'MENTIONS'}, {'related': 'EU lawmakers',
'relationship': 'MENTIONS'}, {'related': 'Community Notes', 'relationship': 'MENTIONS'}, {'related': 'Commission', 'relationship': 'MENTIONS'}, {'related':
'Israel', 'relationship': 'MENTIONS'}, {'related': 'DSA', 'relationship': 'MENTIONS'}, {'related': 'Digital Services Act', 'relationship': 'MENTIONS'},
{'related': 'Thierry Breton', 'relationship': 'MENTIONS'}, {'related': 'transparency obligations', 'relationship': 'MENTIONS'}]
</item 5>

<item 6>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> disinformation
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> disinformation

Entities: disinformation, EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks, Blue Check system, legacy
account verification, illegal content, EU, TechCrunch, Hamas attacks, X, Elon Musk, Hamas, Israel, DSA, Gaza
</item 6>

<item 7>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“In this context, 832 videos related to the war have been fact-checked, of which 211 have been removed,” Jourova noted.

The EU also flagged reporting by Microsoft that told it Bing Search had either promoted information or downgraded questionable information in relation to almost
800,000 search queries related to the Ukraine crisis.

Jourova’s speech also highlighted a couple of other areas where she urged Code signatories to go further — calling (yet again) for more consistent moderation
and investment in fact-checking, especially in smaller Member States and languages.

She also criticized platforms over access to data, saying they must step up efforts to make sure researchers are empowered to scrutinize disinformation flows
“and contribute to the necessary transparency”.

Both are areas where X/Twitter under new owner, Elon Musk, has moved out of step with EU expectations on countering disinformation.

Twitter (now X) was an original signatory to the disinformation Code but Musk took the platform out of the initiative back in May, as critical scrutiny of his
actions dialled up in the EU.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: EU, Microsoft, TechCrunch, X/Twitter, Elon Musk, Jourova, Deepfake election risks trigger EU call for more generative AI safeguards, generative AI
safeguards, Bing Search, Ukraine crisis, disinformation Code
</item 7>

<item 8>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU said the probe will also look at the effectiveness of “related policies mitigating risks to civic discourse and electoral processes”

— with a focus on the effectiveness of its “Community Notes” system of crowdsourcing fact-checking to aggregated views of users, who may vote to label disputed
tweets with additional context. The EU said the probe will also look at the Transparency requirements — with a focus on “suspected shortcomings” by X in giving
researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected shortcomings in its ads repository (aka the ads transparency
library the regulation also mandates)

— with a focus on “suspected shortcomings” by X in giving researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected
shortcomings in its ads repository (aka the ads transparency library the regulation also mandates) Suspected deceptive design elements in X’s user interface —
“notably in relation to checkmarks linked to certain subscription products, the so-called Blue checks”

“If proven,

Graph facts:
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> moderation
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> fact-checking
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> fact-checking
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> moderation
- Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society -MENTIONS-> moderation

Entities: fact-checking, moderation, EU, TechCrunch, deceptive UX design, Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation,
transparency and deceptive UX design, first DSA probe in EU, illegal content risks, civic discourse, electoral processes, Article 40 of the DSA, ads repository,
Blue checks, user interface, subscription products
</item 8>

<item 9>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: EU, content moderation, TechCrunch, deceptive UX design, Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation,
transparency and deceptive UX design, first DSA probe in EU, illegal content risks, X, EU regulators, the Commission, crowdsourced opinions, diverse information
risks, harm exposure, Elon Musk, EU lawmakers
</item 9>

<item 10>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> disinformation
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> disinformation

Entities: disinformation, TechCrunch, X, Elon Musk, EU lawmakers, EU’s top diplomat, Code of Practice on Disinformation, Kremlin propaganda, Twitter APIs,
Commission, X (formerly Twitter) is worst for disinformation, per EU analysis, DSA
</item 10>
=== END ITEM id=28e4d94ab1a4ac9f ===
