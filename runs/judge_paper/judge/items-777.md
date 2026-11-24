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

This file has 4 items.

=== ITEM id=404a111e52c90b8c ===
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
A rule of law clash between Musk and the EU has looked increasingly inevitable in recent months — and, indeed, has been predicted by some industry watchers
since rumors of the erratic billionaire’s plan to take over Twitter emerged last year.

Reminder: Penalties for confirmed breaches of the DSA can be as high as 6% of global annual turnover. The bloc also has powers, in extremis, which could result
in access to X being shut off in the region if the platform repeatedly fails to correct course. So if Musk fails to satisfy EU regulators, he could face serious
consequences for what remains a highly indebted business.

In his letter to Musk, Breton writes that the EU has “indications” that X is being used to disseminate illegal content and disinformation in the EU” following
Saturday’s attacks. He then goes on to remind the company of the DSA’s “very precise obligations” vis-à-vis content moderation.
</item 1>

<item 2>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
The law puts an obligation on so-called very large online platforms (VLOPs) to assess societal risks attached to their use of algorithms — and put in place
“reasonable, proportionate and effective mitigation measures” for identified risks, with their reporting and mitigation plans subject to independent audit and
oversight by the Commission.

We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.

Update: A Commission spokesperson told us:

While the Code is voluntary, compliance with the DSA is not. The DSA is now legally binding on X/Twitter. This includes in particular the obligation of Very
Large Online Platforms, such as X/Twitter to mitigate the risks their platforms pose for the spread of disinformation. The TrustLab study to measure two
Structural Indicators covers signatories of the Code at the time of the launch of the exercise, which at the time included X/Twitter.
</item 2>

<item 3>
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
</item 3>

<item 4>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Reached for a response to the EU probe, X’s Joe Benarroch — who was brought into the company by CEO Linda Yaccarino relatively recently to work on business
operations — said:

X remains committed to complying with the Digital Services Act, and is cooperating with the regulatory process. It is important that this process remains free
of political influence and follows the law. X is focused on creating a safe and inclusive environment for all users on our platform, while protecting freedom of
expression, and we will continue to work tirelessly towards this goal.

While X under Musk has very evidently been flying the ship in the opposite direction to the responsible governance the DSA intends to encourage — hence
attracting so many warnings from EU lawmakers, even long before today’s formal probe — it’s notable X recently started offering a research program for outsiders
to get data to systemic risks arising on the platform in the EU. However the Commission obviously has doubts X has gone far enough on the transparency front to
meet the DSA’s bar.
</item 4>

<item 5>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
In a very swift test of the European Union’s newly updated content moderation rulebook, the bloc has fired a public warning at Elon Musk–owned X (formerly
Twitter) for failing to tackle illegal content circulating on the platform in the wake of Saturday’s deadly attacks on Israel by Hamas terrorists based in the
Gaza Strip.

The European Commission has also raised concerns about the spread of disinformation on X related to the terrorist attacks and their aftermath.

Unlike terrorism content, disinformation is not illegal in the EU per se. However, the EU’s Digital Services Act (DSA) puts an obligation on X — as a so-called
very large online platform — to mitigate risks attached to harmful falsehoods as well as to act diligently on reports of illegal content.

Following the terrorist attacks by Hamas against 🇮🇱, we have indications of X/Twitter being used to disseminate illegal content & disinformation in the EU.
</item 5>

<item 6>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Confirmed breaches of the online governance regime can face a range of major sanctions, including fines of up to 6% of global annual turnover.

The EU can also apply interim measures where it believes there’s a risk of serious harm for users and may even seek to have access to infringing services
temporarily blocked. Although its announcement today is careful to caveat the development by affirming no conclusions have yet been reached.

There is also no confirmed timeline for the probe to conclude. But the fact of it being opened means other tools are put within reach of regulators and could be
applied if the EU sees the need to act quickly — such as the aforementioned interim measures. So the Commission’s official scrutiny of X could have real world
implications for how the platform operates sooner rather than later.
</item 6>

<item 7>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
But at the time of writing, Musk had engaged in a bit of a dance of his own with the news of the EU’s warning by responding to a critical tweet posted to X by
journalist Glenn Greenwald, who attacked the EU’s new “censorship law,” as he dubbed the DSA — which he claimed would be used to “punish X.”

In reply to Greenwald, Musk avoided expressing the same trenchant criticism of the EU’s approach but invited yet more relativism — writing: “Let the public hear
exactly what this disinformation consists of and decide for themselves.”

He then went on to sew doubt that disinformation is something that can be independently arbitrated at all, implying fact-checking is just a convenient exercise
for targeting different opinions — so essentially echoing Greenwald’s position — by claiming that “many times” the “official fact-checker” has been found making
false statements, before adding a rhetorically empty offer that “Maybe this is the case here, maybe not” for good, plausible deniability measure.
</item 7>

<item 8>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” he warns. “We have, from qualified sources, reports about potentially illegal content circulating on your service despite flags from relevant
authorities.”

He also takes issue with a change X made last night to its public interest policy, under which it judges newsworthiness (i.e., for deciding whether posts that
infringe its rules may nonetheless remain on the site) — but which Breton says have left “many European users uncertain” (i.e., about how X is applying its own
rules).

Again, this is a problem because the DSA requires platforms to be clear and transparent about their rules and how they apply them. “This is particularly
relevant when it comes to violent and terrorist content that appears to circulate on your platform,” Breton continues in another pointed warning.
</item 8>

<item 9>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.
</item 9>

<item 10>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
During this period the DSA had not entered into force yet for VLOPs and VLOSEs, including X/Twitter. The Commission cannot comment further on the substance of
the risk assessments. We are analysing the available information on a range of issues and we are in close contact with X but we cannot comment on specifics and
timelines. As stated by Commissioner [Thierry] Breton, minor protection and integrity of elections are priorities in the enforcement of the DSA.

Penalties for breaches of the DSA can scale up to 6% of global annual turnover and the Commission has previously wanted that not following the Code of Practice
on Disinformation would be factored into its assessments. So the price for X ignoring the EU’s warnings on disinformation could be steep — albeit its ad revenue
has cratered since Musk’s takeover, as a result of his failure to tackle toxicity and harmful disinformation encouraging an advertiser exodus — and shrinking
earnings would effectively downsize any future penalties X could face under the DSA.
</item 10>

<item 11>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 11>

<item 12>
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
</item 12>

<item 13>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 13>

<item 14>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.
</item 14>

<item 15>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
It’s official! Elon Musk’s X marks the spot of the first confirmed investigation opened by the European Union under its rebooted digital rulebook, the Digital
Services Act (DSA).

Announcing the opening of a “formal proceeding” today, the European Commission said the investigation will look at whether the social networking platform may
have breached rules linked to risk management, content moderation, dark patterns, advertising transparency and data access for researchers.

The opening of a formal DSA investigation on X comes hard on the heels of a complaint against X’s adtech by privacy rights group, noyb — although today’s formal
proceeding is unlikely to be directly linked as the Commission has been probing the platform for months via a flurry of requests for information. Its earlier
actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.

Back in October the Commission sent Musk’s company an “urgent” formal request for information about how it was responding to information risks arising out of
the conflict.
</item 15>

<item 16>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 16>

<item 17>
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
</item 17>

<item 18>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X has been pushing a so-called Community Notes feature — which crowdsources additional context to add to questionable tweets — as its main response to
disinformation risks.

By Thursday afternoon, Hamas’ attack had killed more than 1,200 people, per the Israeli Military, and at least 1,537 people had been killed in Gaza by Israel’s
retaliatory strikes, according to the Gaza Ministry of Health.

Returning to the EU process, X has until October 18 to provide the bloc with information about “the activation and functioning of X’s crisis response protocol”;
and until October 31 to respond to its other requests.

“Based on the assessment of X replies, the Commission will assess next steps,” the Commission wrote. “This could entail the formal opening of proceedings
pursuant to Article 66 of the DSA.”

It also noted the regulation includes powers to impose fines for “incorrect, incomplete or misleading information in response to a request for information”.

“In case of failure to reply by X, the Commission may decide to request the information by decision.
</item 18>

<item 19>
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
</item 19>

<item 20>
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
</item 20>

<item 21>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
It replied with an automated email reading: “Busy now, please check back later.”

Update: X’s global government affairs account has posted a thread responding to the EU’s criticism in which it claims the data shows other social networking
services saw greater changes in subscriber growth and total growth for pro-Kremlin accounts, and for engagement and volume; it also writes that: “More than 700
unique Community Notes have appeared on posts related to the Ukraine conflict. Through Community Notes’ new ‘Notes on Media’ feature, which automatically
matches notes on an image or video to other posts that contain matching media, these notes appeared on over 1,200 additional posts. These notes are in numerous
languages and are written specifically for local audiences.”

Back in April, X/Twitter was named as one of 19 larger platforms that face the strictest level of regulation under the EU’s DSA.
</item 21>

<item 22>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
He’s also ripped up a bunch of legacy content moderation policies and slashed in-house enforcement teams while promoting a decentralized, crowdsourced
alternative (rebranded as Community Notes), which essentially outsources responsibility for dealing with tricky issues like disinformation to users in what
looks suspiciously like another gambit to eke out extra engagement and farm confusion by applying a philosophy of extreme relativism so culture warriors are
encouraged to keep forever fighting for their own “truth” in the comments.

Oh and he also pulled X out of the EU’s Code of Practice on Disinformation earlier this year in a very clear thumb of the nose to EU regulators.

Urgent letter to Musk

In an “urgent” letter to Musk today, which the EU’s internal market commissioner Thierry Breton also shared on X, the bloc has sent the strongest signal yet it
believes Musk’s platform is in breach of the DSA — although this is by no means Musk’s first warning.
</item 22>

<item 23>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Subscription revenues to a revamped Twitter Blue paid offering (now called X Premium) don’t appear to have been substantial enough to plug the ad gap — which
may explain Musk’s recent suggestion he plans to charge all users a small fee. However ending free access to X could be the final nail in the coffin of the
platform as a real-time information network if it triggers a further/final exodus of remaining legacy users whose knowledge and engagement lent Twitter its
original value.

Disinformation and spammers may rush in to fill the gap, generating a parody of activity, but without a critical mass of genuine users to pay attention and
contribute value and vitality — aka the life-blood of a bona fide information network — X really will mark the spot of Twitter’s demise.

Also today, a report by misinformation watcher NewsGuard found that a decision by Musk to remove labels denoting state-run or government-affiliated media
accounts led to a surge in propaganda spreading on X, with Russia’s RT seeing the biggest gains per the analysis.

From the report:
</item 23>

<item 24>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
We also clarified the scale of financial penalties possible under the DSA if a breach is confirmed; and added details of the regulation’s crisis response
mechanism. We also fixed an error in the original report which referred to Interpol, rather than Europol.
</item 24>

<item 25>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In this case, failure to reply by the deadline could lead to the imposition of period penalties,” the EU added.

Earlier this year, Musk pulled out of the EU’s Code of Practice on online disinformation. In a response to X leaving the voluntary Code, Breton warned:
“Obligations remain. You can run but you can’t hide.”

In recent days, TikTok and Meta have also been warned by the EU over disinformation related to the Israel-Hamas war. But back in September, a study commissioned
via the EU’s Code of Practice suggested X is the worst of the major platforms when it comes to spreading disinformation.

This report was substantially revised to clarify a number of aspects, including — most saliently — that the EU has sent a formal request for information to X at
this point. This could prefigure a formal announcement of an investigation but is not, yet, that formal technical step.
</item 25>

<item 26>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“I remind you that following the opening of a potential investigation and a finding of non-compliance, penalties can be imposed,” Breton adds, concluding the
letter.

We’ve asked the Commission to confirm whether it has opened an investigation into X’s DSA compliance over concerns it’s raised in the letter. Perhaps it’s going
to wait a day to see what his response is before taking that next dance step.

Musk’s extremely iterative — or just plain erratic/arbitrary — management style is so far from the responsive and responsible qualities the EU’s rulebook
demands of digital leaders it’s hard to see how this clash can end well for either side.

We contacted X for a response to the EU’s warnings about its DSA compliance but at press time the company had not responded — beyond firing back its usual
automated reply, which reads: “Busy now, please check back later.”
</item 26>

<item 27>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In response to the EU’s warning earlier this week, X CEO Linda Yaccarino released a letter saying that a leadership group had been convened to consider X’s
response and “tens of thousands” of pieces of content had been removed, along with “thousands” of posts and “hundreds” of accounts linked to terrorist groups,
violence or extremism.

She also said the company formerly known as Twitter is responding to law enforcement requests but claimed it had not received any requests from Europol at the
time.

Today @lindayaX responded to @ThierryBreton’s letter detailing our work in response to the terrorist attack on Israel. pic.twitter.com/yZtaOVGpHG — Global
Government Affairs (@GlobalAffairs) October 12, 2023

In further remarks via X’s safety account, the company reported that there had been more than 50 million posts globally over the past two days that referenced
the weekend’s terrorist attack, underlining the scope of content generated.
</item 27>

<item 28>
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
</item 28>

<item 29>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 29>

<item 30>
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
</item 30>
=== END ITEM id=404a111e52c90b8c ===

=== ITEM id=5510b9eb3a678237 ===
Gold sentences and facts:
Sentence 0: 9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.
  Fact 0.0: At 9:47 p.m., the Bears answer right back and sack Dobbs.
  Fact 0.1: Sweat and Brisker are in there to take Dobbs down.
Sentence 1: The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in points (16.7) and 10th in yardage (310 per game).
  Fact 1.0: The Ravens defense has allowed the fewest points per game this season, 16.1.
  Fact 1.1: The Ravens defense has allowed the second-fewest total yards this season, 4,030.
  Fact 1.2: San Francisco's defense ranks second in points, at 16.7.
  Fact 1.3: San Francisco's defense ranks 10th in yardage, at 310 yards per game.

Retrieved items:
<item 1>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
The defense can't tolerate those mistakes with all of its injuries.

16. Las Vegas Raiders 5-5 (22)

The Raiders are hot, having won two consecutive games for interim coach Antonio Pierce, bringing up flashbacks to the playoff push they made with Rich Bisaccia
in 2021. They are right back in the AFC wild-card race with the Dolphins, Chiefs, Vikings ahead.

17. New Orleans Saints 5-5 (15)

The Saints will go into a bye not knowing what do at quarterback. with Derek Carr further hurting his right shoulder and needed to be checked for a concussion.
Jameis Winston did provide a spark, too little too late at Minnesota. But the division lead is precarious with poor QB play overall.

19. Indianapolis Colts 5-5 (19)

The Colts stopped scoring at least 20 points in every game during their ugly, run-heavy victory in Germany vs. the Patriots.
</item 1>

<item 2>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They also seemed invigorated for Jonathan Gannon on defense and special teams. Look for the Cardinals to finish strong and play a whole lot of spoiler.

30. New England Patriots 2-8 (28)

The Patriots tried to win a game by hiding Mac Jones and their quarterbacks in general. They almost pulled it off with the running game dominating and
short-handed defense making a lot of plays, but they didn't see the results in Germany ahead of a tough bye for Bill Belichick.

31. New York Giants 2-8 (30)

The Giants can just flush away this season without Daniel Jones and Tyrod Taylor. The injuries go to their offense early and remained relentless, and now their
defense is falling apart, too. The question is, will Brian Daboll get a massive mulligan after his rookie playoff leadership?

32.
</item 2>

<item 3>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9. Jacksonville Jaguars 6-3 (4)

The Jaguars got an NFC wake-up call against the 49ers, who were much more prepared after a Week 9 bye than they were. Jacksonville needs to forget that debacle
and get back on track, stat. The team gets a division double vs. Tennessee and Houston before facing Cincinnati.

10. Dallas Cowboys 6-3 (11)

The Cowboys took advantage of their "no contest" vs. the Giants at home with Dak Prescott continuing to ball as a passer and runner. They need to keep the
defense and running game success, too, in bigger games going forward.

MORE: Inside the Cowboys historic offensive explosion vs. Giants

11.
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But there's something else -- Love was pass rush ignorant, meaning he didn't let the Lions' pressure scare him, even if it did wind up overwhelming him for some
early-game sacks.

RAIDERS: Deserve credit for trying -- in the past two weeks they lead the NFL in pass rush pressure rate and are 10th in blitz rate. Playing weaker offensive
lines in Pittsburgh and L.A. have helped boost those numbers, but quarterbacks are still finding success to the tune of 13.9 yards per catch and 7.07
YAC/reception over those two games, both bottom-three in the league. And, both quarterbacks the Raiders played in Weeks 3 and 4 each posted at least 22 Fantasy
points (Kenny Pickett, Justin Herbert without his second-best receiver and without his top running back).

RAIDERS: The hunch is the Raiders will keep bringing the blitz and play a lot of gutsy man coverage behind it.
</item 4>

<item 5>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
MORE: Inside the Cowboys historic offensive explosion vs. Giants

11. Seattle Seahawks 6-3 (13)

The Seahawks needed every ounce of their offense around Geno Smith to rally to beat the Commanders, but the get-well performance worked well for him, Kenneth
Walker and the wide receivers. The defense could be more consistent.

12. Minnesota Vikings 6-4 (14)

The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running
game for Kevin O'Connell and the defense overall under Brian Flores. Their 0-3 and 1-4 starts are distant memories as the Vikings look for a sixth straight win
in Week 11 vs. Denver.

13. Houston Texans 5-4 (16)

The Texans are looking great with C.J. Stroud leading their offense and now they've found some running game, too.
</item 5>

<item 6>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
32. Carolina Panthers 1-8 (31)

The Panthers found better defense and special teams to keep them in the game in Chicago, but Bryce Young still can't unlock a limited passing game and the
offensive line and running game hasn't help nearly enough.
</item 6>

<item 7>
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
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 8>

<item 9>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
The Ravens would win that spread bet because the margin of victory (0) was narrower than 1.5.

Related Pages:
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Viewers in Canada can stream every NFL game via DAZN.

MORE: How Joshua Dobbs got his 'Passtronaut' moniker

Bears vs. Vikings start time

Date: Monday, November 27

Monday, November 27 Start time: 8:15 p.m. ET

Monday's game between the Bears and Vikings will kick off at 8:15 p.m. ET (7:15 p.m. local time) from U.S. Bank Stadium in Minneapolis, Minn.

The Bears haven't beaten the Vikings since 2020, with Minnesota 6-1 against Chicago over the last four seasons. The Vikings are 8-3 at home against the Vikings
since 2012.
</item 10>

<item 11>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
DeVito has thrown for 858 yards, seven touchdowns, and one interception in five games since being named New York's starting quarterback, posting the
13th-highest passer rating in the NFL during that stretch (97.3).

Baltimore Ravens (11-3) at San Francisco 49ers (11-3) | 8:15 p.m. ET

Potential Super Bowl sneak peek and MVP implications: Easily the most hyped game of the week, with the top seed in each respective conference squaring off in
what could be a potential Super Bowl LVIII preview as well as the decisive clash for the MVP Award.

According to oddschecker, 49ers QB Brock Purdy is listed as the heavy favorite to be named league MVP (-230), while Baltimore signal-caller Lamar Jackson holds
the second-shortest odds (+525).

Purdy, the final pick of the 2022 NFL Draft, has seen his career unfold as if it were a fairy tale that just keeps getting more remarkable.
</item 11>

<item 12>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their defense should be getting a little better to support their often explosive offense. They should feel good about a Week 11 rebound vs. the Raiders.

7. Pittsburgh Steelers 6-3 (9)

The Steelers are playing the way Mike Tomlin wants, with the running game rolling to big totals with Jaylen Warren and Najee Harris and the defense doing what
it needs, more so vs. pass than run, in key moments. You can't argue with the overachieving results.

8. Cleveland Browns 6-3 (10)

The Browns proved they can win games with defense in another way, with big plays. They also have more offensive pop when Deshaun Watson plays well and the
running game doesn't slump. Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9.
</item 12>

<item 13>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
3.

We have two sets of teammates in our top 10 this week: Detroit's David Montgomery and Jahmyr Gibbs (@ Bears) and Miami's Raheem Mostert and De'Von Achane (vs.
Titans). Both duos have such solid workloads in high-flying, run-first offenses — and such juicy matchups — that we can't imagine any of these studs finishing
outside the elite eight. DMont and Gibbs combined for 193 yards and two TDs when they faced Chicago in Week 11. Tennessee just surrendered 31 points to Indy, so
we know Miami will be dominating the time of possession and running early and often against Will Levis and company.

Jets fans will be hoping for some Breece Lightning from stud second-year back Breece Hall (vs. Texans), and we think they will be rewarded for their patience.
</item 13>

<item 14>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
8:09 p.m. — Perhaps the most interesting storyline tonight will be whether Al Michaels can keep a brave face throughout what might not be the most appealing
game.

8:00 p.m. — We're going to keep spirits high tonight despite a matchup of teams with a combined three wins. All football is good football! Let's see how long
this mentality lasts...

7:46 p.m. — It's 52 degrees and clear in Chicago right now, so certainly not as cold as it could be in the Windy City this time of year. Still, it's a rare
colder-weather game for Bryce Young.

7:30 p.m. — The Bears will be donning orange tonight, including an alternate orange helmet.

7:10 p.m. — D.J. Moore is back facing his former team tonight, eight months after he was traded to the Bears in the deal that sent the No. 1 pick to Carolina.

6:50 p.m.
</item 14>

<item 15>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
</item 15>

<item 16>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 16>

<item 17>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They need Matthew Stafford to help save face the rest of the way or the season will look a lot like 2022.

27. Tennessee Titans 3-6 (27)

Will Levis is going through rookie lumps after his dazzling debut as the brutal offensive line woes exacerbated by injuries are getting to him and the entire
Titans offense. The Titans still need to keep starting him, because they can't overcome those systemic problems or those in overall pass defense.

28. Chicago Bears 3-7 (29)

The Bears don't care how they win, but they should be pleased their running game and defense bailed out their limited passing game with Tyson Bagent to secure
another win in Week 10. They will hope to finish strong with Justin Fields soon.

29. Arizona Cardinals 2-8 (32)

The Cardinals got an immediate spark for their offense with Kyler Murray and James Conner back.
</item 17>

<item 18>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
The Panthers have allowed the most rushing TDs (14), the third-most points per game (28.4), and the fourth-most rushing yards per game (139.4) during their 1-7
start. The Colts will cruise in this one.

FINAL SCORE PREDICTION: Colts 27, Panthers 17

Seahawks (+6) at Ravens

Baltimore's defense has been incredible of late, and the Ravens seem to be flying under the radar (so to speak) at 6-2. However, the sportsbooks have clearly
taken note of their awesome run, giving the NFC West-leading Seahawks a full touchdown on the road. This seems like a great opportunity for us to rock with the
'dog, considering Seattle has won two straight contests and gone 2-1 on the road (including a big win over the high-flying Lions in Detroit).
</item 18>

<item 19>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Rams) and think he has the potential to smash Los Angeles at home this week. The speedster has quietly racked up over 300 yards and two TDs over the Ravens'
past four games, and with Lamar Jackson facing Aaron Donald and a still-talented pass-rush, John Harbaugh's squad might turn to the rookie to move the chains
between the 20s. All we need is one or two big chunk gains to make him a worthwhile start — he can run friggin' 22 miles per hour!

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy RB busts in Week 14?

While he just barely cracks our top 25 thanks to his passing-down usage, James Cook (@ Chiefs) doesn't thrill us at Arrowhead against the pissed-off defending
champs.
</item 19>

<item 20>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:25 p.m. — The rookie Ivan Pace Jr. finally gets to Fields and drops him for a five-yard loss. It'll be 3rd & long for the Bears.

8:24 p.m. — Fields throws on the run to find Roschon Johnson for a first down, and he quickly follows up with a 13-yard pass to Khalil Herbert. The Bears are
driving, and Fields doesn't have an incompletion yet.

8:21 p.m. — The Bears have taken it into Vikings territory with a string of short gains. For Chicago, that's progress.

8:19 p.m. — Fields finds Moore for a short gain and the first 1st down of the night.

8:16 p.m. — The Bears will start with the ball on the road, looking to build early momentum.

8:00 p.m. — Will the Josh Dobbs show continue tonight?
</item 20>

<item 21>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 21>

<item 22>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
BEARS: Are one of the easiest matchups for a quarterback. They rank in the bottom-seven in completion rate allowed (70.1%), yards per catch allowed (12.2),
Yards After Catch per reception (YAC/reception) allowed (6.26, second-most) and missed tackles (52). One more factor: Chicago ranks poorly in pass rush pressure
rate, blitz rate and has just two sacks all season.

HOWELL: When not pressured has completed 73.8% of his throws for 7.2 yards per attempt with a 9.7% off-target rate. None of these are particularly impressive --
they're all about league-average -- but that's still a plus for Howell considering this week's matchup.

PASS ATTEMPTS: Through four weeks the Bears are seeing an average of 31.8 pass attempts against them per game. Even in a game the Bears led for much of last
week, the Broncos tried just 28 throws.
</item 22>

<item 23>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can QB Jalen Hurts and the Eagles offense finally bounce back?: Philadelphia's defense isn't the only unit that needs to rebound, as the offense has struggled
mightily over the last three games, with Hurts' ineffectiveness playing a significant role. After averaging the third-most points in the NFL over the first 11
games (28.2), the Eagles have scored four offensive touchdowns during their current skid, three of which came via the "tush push."

The Eagles may have difficulty throwing against a solid Giants pass defense, though they could alleviate some of their offensive woes by running the ball
efficiently. New York gives up 131.6 rushing yards per game, the fourth-most in the league.

Does Giants QB Tommy DeVito have any magic left?: The Giants may have a terrible offense, but this unit has operated at its best under DeVito, an undrafted
rookie and native of Cedar Grove, New Jersey.
</item 23>

<item 24>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.
</item 24>

<item 25>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 25>

<item 26>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Could this be the weekend that Keaton Mitchell (vs. Rams) gets back into RB2 range for the Ravens? Los Angeles might find itself spying Lamar Jackson and Gus
Edwards so often that the speedy rookie busts a couple of big runs. Mitchell has racked up 330 yards and two TDs since he broke out four games ago, but the
majority of that damage (134 yards and a TD) came in the initial breakout against Seattle. He has still yet to see 12 touches in a game, something we can't
imagine will be the case by the end of the regular season.

We also think Roschon Johnson (vs. Lions) could have a chance to take over lead-back duties in Chicago, as both Khalil Herbert and D'Onta Foreman have battled
injuries and inconsistent touches this season.
</item 26>

<item 27>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Stroud leading their offense and now they've found some running game, too. The defense is just making enough plays to put them in position to win high-scoring
games. Houston is a legitimate playoff contender under rookie coach DeMeco Ryans.

14. Cincinnati Bengals 5-4 (7)

The Bengals are battling some key injuries, leaving them much less room for error. Unfortunately, Joe Burrow made key mistakes and the defense wilted against
run and pass at home in the loss to the Texans.

15. Buffalo Bills 5-5 (12)

The Bills fell right into the trap game against the Broncos at home, losing because of more mistakes from Josh Allen. The defense can't tolerate those mistakes
with all of its injuries.

16.
</item 27>

<item 28>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.
</item 28>

<item 29>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Buccaneers (DK: $3,000 | FD: $4,700 | SD: 1.8x)

NFL DFS Picks Week 15: D/ST sleepers, values for DraftKings, FanDuel

Los Angeles Rams vs. Commanders (DK: $3,100 | FD: $4,300)

The Rams got destroyed by MVP candidate Lamar Jackson and the high-flying Ravens in Week 14, but push that out of your memory banks. Before that game, Sean
McVay's squad had surrendered 20 or fewer points in four straight contests. Aaron Donald and company also have 17 sacks and six takeaways over their past six
games, so they'll be licking their chops looking across the line at Sam Howell this weekend. No QB has more interceptions than Howell, who has also taken 10
more sacks than any other signal-caller this season.
</item 29>

<item 30>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.
</item 30>

<item 31>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 31>

<item 32>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
That just hung up there. The Bears have been all over Dobbs and the Vikings' offense early on. A taunting penalty after the play hurts Chicago's field position.

Bears 3, Vikings 0

8:44 p.m. FIELD GOAL — The drive fizzles out, and the Bears opt to kick the field goal on 4th & 3. It's good. The Bears are out to an early 3-0 lead, but
they've left plenty of points on the board.

End of first quarter: Bears 0, Vikings 0

8:39 p.m. — What a play by Fields to get the ball to a wide open Kmet! The Vikings brought intense pressure, allowing Kmet to get wide open and go for 24 yards.
The Bears are in business.

8:38 p.m. — 2nd & 22 becomes 2nd & 27, but the Bears make up 17 yards on two plays.
</item 32>

<item 33>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Patriots. At least Jonathan Taylor is doing his normal things and the defense is playing well again while Gardner Minshew looks meh.

18. Los Angeles Chargers 4-5 (17)

The Chargers got well offensively with Justin Herbert, Keenan Allen and Austin Ekeler against the Lions. But the defense went back to crumbling vs. big pass
plays and added wilting against the run. The pressure is right back on Brandon Staley with a trip to Green Bay next.

20. Tampa Bay Buccaneers 4-5 (23)

Baker Mayfield and the offense is back on track to move the ball well after a brief midseason slump. They're still not lighting it up, but the Bucs are back to
playing smart complementary football with their defense rebounding and still might have found a long-term QB answer.

21.
</item 33>

<item 34>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Week 10 in the NFL had some of the league's best teams not playing. But that didn't stop the excitement, with five teams winning on walk-off field goals and
more games coming right down to the wire.

With that, there were some surprising results and expected domination. Some favorites dug deep to win. Some underdogs turned in shocking upsets.

Putting the latest wins and losses into a blender, here's Sporting News' updated look at how all teams stack up against each other, 1-32 going into Week 11:

NFL WEEK 11 PICKS: Straight up | Against the spread

NFL power rankings Week 11

1. Philadelphia Eagles 8-1 (previous week: 1)

The Eagles enjoyed a bye trying to get a little healthier for the more difficult stretch run in a top-heavy NFC as they have tough games at the Chiefs (the
Super Bowl 57 rematch), vs. the Bills, vs.
</item 34>

<item 35>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Can't say the same against the run as the Redbirds rank in the bottom five in pretty much every single run defense metric except yards after contact per rush
(where they're league average). A running back has scored 17 or more PPR points in four of their past five against the Cardinals, and eight in 12 games have hit
at least 15 PPR points. With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.
</item 35>

<item 36>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
It's 4th & 10, and Chicago is lining up to got for it.

8:36 p.m. — Sacks are killing the Bears in this game. Multiple Vikings defenders wrap up Fields for a 12-yard loss and quickly set Chicago back to 2nd & 22.

8:35 p.m. — The Bears' offense is picking up where it left off. After Fields takes off for 14 yards, he hits Roschon Johnson for 22 yards to take Chicago into
Vikings territory.

8:31 p.m. — That's what the Bears got Montez Sweat to do. Sweat sacks Dobbs on third down to force a quick three-and-out and get the ball right back in Fields'
hands.

8:27 p.m. — A great start for the Bears ends with zero points as Cairo Santos misses a 48-yard field goal. That's a deflating sequence for Chicago.

8:25 p.m. — The rookie Ivan Pace Jr.
</item 36>

<item 37>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Carolina has serious questions to answer.

10:05 p.m. — Young gets a few yards back but is sacked by Yannick Ngakoue on third down. The Panthers will punt again.

Yannick Ngakoue gets his team off the field on 3rd down 😤#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/JPt1xc3b93 — NFL (@NFL) November 10, 2023

10:04 p.m. — Bryce Young is now lining up in his own end zone after Miles Sanders was dropped for a six-yard loss.

10:01 p.m. — Punt. Again.

9:57 p.m. — The Panthers get a grand total of one yard (-4 including a delay of game penalty) and punt it away less than a minute into the first half.

End of first half: Panthers 10, Bears 9

9:40 p.m.
</item 37>

<item 38>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— And now this one is officially over. The Bears pick up their third win of the season, while Carolina drops to 1-8 and faces more tough questions about Bryce
Young and the offense.

11:08 p.m. — Bagent hits Mooney for the first down on 3rd & 7, and the Panthers now won't be able to stop the clock.

11:06 p.m. — Pineiro's attempt is SHORT from 59 yards out. The gamble fails for Frank Reich. The Panthers will hope to get the ball back.

11:04 p.m. — Young is very nearly intercepted on 3rd & 10, and the Panthers say they've seen enough of the offense. Eddy Pineiro is attempting a 59-yard field
goal to tie the game.

11:00 p.m. — Miles Sanders gets himself wide open on fourth down to extend the drive.
</item 38>

<item 39>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.
</item 39>

<item 40>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
His off-target rate in the fourth was 9.1%, which was closer to what he had averaged in his first three games (13.6%).

BRONCOS: Have allowed season-best Fantasy numbers to each of the past three quarterbacks they've played this season. They're a terrific matchup for anybody
right now. Remember this when we talk about the Jets.

COMMANDERS: Sport a slightly below-average pass rush pressure rate thanks to only one game all year with a rate over 30% (52.7% at Denver, and Wilson still had
a great game even if you take away a Hail Mary touchdown catch). They rank seventh in sacks (13) and 13h in pressures (53), which again floats around league
average.

BEARS: Rank league-worst in pass rush pressure rate allowed at 47.4% of their snaps -- nearly one out of every two drop backs is pressured.
</item 40>

<item 41>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 41>

<item 42>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
21. New York Jets 4-5 (18)

Aaron Rodgers can't come back soon enough, and going for a December miracle might be too late after two straight key losses to the Chargers and Raiders. Zach
Wilson will always cap the potential of a talented offense that keeps making mistakes. The Bills and Dolphins in back-to-back division games next might be their
season.

22. Denver Broncos 4-5 (24)

The Broncos got another big win against a AFC power, sandwiching their Week 9 bye with upsets over the Chiefs and Bills. The offense has hit a balanced grooved
with Russell Wilson and the defense is making a lot more plays, too.

23. Washington Commanders 4-5 (21)

The Commanders should be pleased about Sam Howell and the offense showing more fight with additional weapons and the running backs look better all-around again.
But the reeling defense and pass protection continue to be un-winnable assets.

24.
</item 42>

<item 43>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
With the Ravens, Bills, Vikings, Raiders, Giants, and Bears idle in Week 13, that means Davante Adams, Stefon Diggs, Justin Jefferson, DJ Moore, Zay Flowers,
Jakobi Meyers, and Jordan Addison are among the most notable wideouts omitted from our rankings. Deep-league owners will also be without Odell Beckham Jr., K.J.
Osborn, Gabe Davis, Jalin Hyatt, and Khalil Shakir.

WEEK 13 PPR RANKINGS:

QBs | RBs | TEs | D/STs | Kickers

Unfortunately, we also won't get to pick on the poor WR defenses of the Giants and Vikings, but it's a relief to see the Ravens' shut-down secondary on the
sidelines.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy WRs in Week 13?
</item 43>

<item 44>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears and Panthers entered Thursday night looking for progress, not playoff contention. If any team achieved that goal, it was Chicago.

It wasn't an aesthetically pleasing game, but that was never the expectation. The Bears grinded out a 16-13 victory behind a terrific defensive performance, a
solid running game, and a few timely throws by Tyson Bagent.

The Panthers jumped out to an early lead on a punt returned for a touchdown by Ihmir Smith-Marsette, but they never found the end zone again. The only offensive
touchdown of the night came in the third quarter, when D'Onta Foreman rumbled his way into the end zone to give Chicago a 16-10 lead.

NFL WEEK 10 PICKS: Straight up | Against the spread

As it turned out, 16 points was all the Bears needed.
</item 44>

<item 45>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:04 p.m. — Dobbs is picked off again! That's two interceptions and just nine passing yards for the Vikings quarterback, but this one doesn't look like his
fault - Jordan Addison saw it bounce off his hands. It's Jaquan Brisker with the pick.

9:01 p.m. — Another penalty gifts the Vikings a first down... Both teams are piling on the mistakes.

8:58 p.m. — A roughing-the-passer call goes against the Bears and gives the Vikings some breathing room early in the drive.

8:54 p.m. — Fields overthrows Darnell Mooney on third down, and Mooney gets destroyed in a collision. The Vikings will get the ball back, still somehow only
trailing by three. Fields will want that one back.

8:50 p.m. INTERCEPTION — Jaylon Johnson intercepts a throw from Dobbs! That just hung up there.
</item 45>

<item 46>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.
</item 46>

<item 47>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Cover 7 | Monday A daily NFL destination that provides in-depth analysis of football’s biggest stories. Each Monday, Mike Sando breaks down the six most
impactful takeaways from the week.

The San Francisco 49ers entered Week 6 riding a 15-game winning streak and looking unstoppable with a young quarterback, Brock Purdy, who was making the game
look easy. They exited with a 19-17 defeat at Cleveland, injuries to key players and, for the first time this season, a question mark or two.

Advertisement

With the previously unbeaten Philadelphia Eagles also losing ugly Sunday, this edition of the Pick Six column addresses key questions facing Super Bowl
favorites with one-third of the 18-week regular season nearly complete.

So far this season, we’ve seen the Miami Dolphins set NFL records for offense, only to lose by four touchdowns in their lone game against a true contender.
We’ve seen the Kansas City Chiefs win with defense, sometimes in spite of their offense. We’ve seen the Buffalo Bills incur devastating injuries.
</item 47>

<item 48>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He also ducked under an end-zone target instead of leaping for it. It's worth noting that a 36-yard crosser he ran against the Chargers flies in the face of
this as he separated from man coverage with speed and technique, then turned upfield and ran outside before racing past a defender and diving for the pylon.

OSBORN: Has a 50% catch rate on the year and that includes whether he faces zone coverage or man coverage. He sees more targets per route run against man
coverage and averages more yards per catch and yards after catch against man coverage. Over his career, targets do seem to come more frequently against man
coverage and he has been more explosive with them.

BEARS: Play zone coverage at the fourth-highest rate of any defense this season (79.2%). And because Cousins is much more explosive of a passer against man
coverage, expect Chicago to keep up their zone defense.
</item 48>

<item 49>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
He goes 18 yards here to put Minnesota near midfield.

10:29 p.m. FUMBLE — That's exactly what CAN'T happen if you're the Bears. Fields loses the football, Sheldon Day recovers, and the Bears waste an opportunity to
make this a two-score game.

Josh Metellus forces a fumble and the Vikings take over 😤



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/fKnE0eTxsP — NFL (@NFL) November 28, 2023

10:27 p.m. — Fields finds Mooney for a big third down conversion as the Bears get firmly into field goal range.

10:25 p.m. INTERCEPTION — Josh Dobbs throws his FOURTH interception of the night. Justin Jones nearly had it, but it was Kyler Gordon who came away with it.
</item 49>

<item 50>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
He's already missed from 48 tonight.

End of third quarter: Bears 6, Vikings 3

10:14 p.m. — The Bears are starting to look respectable on third down again, this time completing to D.J. Moore to move deeper into Vikings territory. Chicago
still might need another first down to give Cairo Santos a fair shot at a field goal.

10:08 p.m. INTERCEPTION — Dobbs throws his third interception of the night! Jaylon Johnson deflected it, Edwards caught it, and interestingly, the Bears have
less favorable field position than they would've if it had fallen incomplete.

And Edwards follows it up with the Bears 3rd INT of the game ‼️
</item 50>
=== END ITEM id=5510b9eb3a678237 ===

=== ITEM id=829d0e0f8cdb7842 ===
Gold sentences and facts:
Sentence 0: That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.
  Fact 0.0: The answer ultimately lies with the jury.
  Fact 0.1: The jury was selected on Wednesday morning.
  Fact 0.2: The jury was selected before lawyers for the government.
  Fact 0.3: Bankman-Fried then swapped two very different stories of the former crypto mogul's sudden rise and almost instantaneous fall.
Sentence 1: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 1.0: The highly anticipated criminal trial of Sam Bankman-Fried started on Tuesday.
  Fact 1.1: Sam Bankman-Fried is the former CEO of bankrupt crypto exchange FTX.
  Fact 1.2: The trial is to determine whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 2>

<item 3>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“All of that was built on lies,” Rehn said.

In his opening statement, Rehn dodged explaining cryptocurrency to the jury. Instead, he punched hard on Bankman-Fried lying and stealing.

Bankman-Fried sat almost motionless, occasionally glancing at Rehn, as the prosecutor told the jury that Bankman-Fried sold stock in FTX and borrowed millions
from lenders by lying.

The story Rehn told is familiar to anyone following the news. In May and June of 2022, Alameda Research — the crypto trading company ostensibly helmed by
Caroline Ellison — didn’t have enough to pay its bills, so it pulled customer money to repay loans. By September, the hole in the FTX balance sheet was so big
that customers could never be repaid.

FTX “didn’t have a chief risk officer, which became an issue when the storm hit.”

When CoinDesk published its article in November 2022, people realized FTX was a house of cards, Rehn said. Meanwhile, Bankman-Fried tweeted.
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 4>

<item 5>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 5>

<item 6>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 7>

<item 8>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
In it, Bankman-Fried tried to justify the missing funds through the margin loan program. It was such a flimsy excuse that even George Stephanopoulos could see
through it.

Did Bankman-Fried really think he could get away with it all? He must still think so, or he would not be dragging his closest companions’ names through the mud
in a jury trial. Throughout the entire prosecution’s case, the defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison.
The hits the defense made on Singh didn’t make me doubt the overall truth of his story — they just made me think that Singh was kind of self-serving and trying
to secure himself a lenient sentence.
</item 8>

<item 9>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Follow along for all the latest news and regular updates from the trial.

But a lot of testimony relied on Ellison recounting conversations she’d had in person, or on auto-deleting text messaging platforms. This gave the defense an
opportunity to try to make her sound unreliable. After all, she had an incentive to flip on Bankman-Fried: the possibility of leniency in her sentencing. Given
her fun tweets about speed, the fact that she was Bankman-Fried’s ex-girlfriend, and that she’d apparently written a bunch of stuff down, I was expecting
fireworks. For the first time in this trial, maybe the defense had an opening.

Instead, I got a sad trombone. In Cohen’s disorganized cross-examination, he mostly bored the jury. At one point, two different jurors appeared to be asleep.

Midway through the morning I began wondering if there was a mercy rule for cross-examinations. Prosecutor Danielle Sassoon had run an effective direct
examination, creating an easy-to-follow narrative.
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 10>

<item 11>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.
</item 11>

<item 12>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 12>

<item 13>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 13>

<item 14>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Follow along for all the latest news and regular updates from the trial.

Yesterday, during an evidentiary hearing, Bankman-Fried was repeatedly scolded by Judge Lewis Kaplan for not answering prosecutor Danielle Sassoon’s questions
on cross-examination. Today, Bankman-Fried was scolded by Kaplan for not answering his counsel Mark Cohen’s questions on direct examination. Bankman-Fried has
also occasionally interrupted Cohen with “yes” and “yup.”

We spent a lot of the morning explaining vocabulary. I will spare you the full list, though I will say that explaining “Amazon Web Services” and “database” was
a bit too detail-oriented. Then Bankman-Fried tried to define “market manipulation.” After Bankman-Fried gave his definition, Kaplan told the jury that he was
the final authority on that, thanks.

In fairness to Bankman-Fried, he has been clearer and much easier to understand than he was at the evidentiary hearing yesterday. There was a minimum of word
salad today.
</item 14>

<item 15>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 15>

<item 16>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 16>

<item 17>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 17>

<item 18>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I was ready to hear this pursued further — but then it, too, was dropped.

I think the defense was also trying to suggest that the government had coerced Ellison’s testimony, by suggesting that she had pleaded guilty to a charge of
defrauding investors that she couldn’t have been involved in. After all, she didn’t prepare materials for them. Unfortunately, she did say that she had
conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine. This
line of questioning felt like a waste of time.

There were rather a lot of sidebars during the cross-examination, to the point that when one occurred, several jurors looked entertained. There were a few
yesterday, too, including one in which the prosecution complained that Bankman-Fried was visibly scoffing at Ellison’s answers, according to the court
transcript.
</item 18>

<item 19>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“I was concerned that I had unwittingly written code that contributed to a crime,” he said.

Soon, however, the clock neared 4:30 p.m., and court adjourned for the day. Yedidia will continue his testimony on Thursday, followed by Matt Huang, a former
partner at the high-powered venture capital firm Sequoia Capital, and then Gary Wang, a key Bankman-Fried lieutenant and one of the government’s star witnesses.
</item 19>

<item 20>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
It is honestly kind of incredible to watch a man torpedo his own credibility on direct testimony. We’re not even at the cross yet, and the judge has already
instructed him to answer the question he’s being asked by his own lawyer.

The jury is watching all of this intently.

The main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his own voice. So far, the count of “Objection,
narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.

Also, sometimes when Bankman-Fried says “we,” he only means himself.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including charges of wire fraud. FTX was a fraud
“from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer
funds.” Follow along for all the latest news and regular updates from the trial.
</item 20>
=== END ITEM id=829d0e0f8cdb7842 ===

=== ITEM id=114bbc9a0b6ced0f ===
Gold sentences and facts:
Sentence 0: As an added bonus, Harris is also the preferred goal-line back, scoring in three of those games.
  Fact 0.0: Harris is also the preferred goal-line back.
  Fact 0.1: Harris scored in three of those games.
Sentence 1: The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by running back Jonah Coleman.
  Fact 1.0: The Wildcats scored a touchdown on their first drive.
  Fact 1.1: The Wildcats' first drive was a seven play drive.
  Fact 1.2: The Wildcats' first drive covered 61 yards.
  Fact 1.3: The Wildcats' first-drive touchdown came thanks to a 25-yard dash.
  Fact 1.4: The 25-yard dash was followed by a dive into the end zone.
  Fact 1.5: Running back Jonah Coleman dove into the end zone.

Retrieved items:
<item 1>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least all of them went to his wideouts. All were in the red zone, none were on deep throws and none were on WR screens.

Sit Him (Lineup Decisions) Projections powered by Sportsline Najee Harris RB PIT Pittsburgh • #22

Age: 25 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs BAL BAL -4 O/U 38 OPP VS RB 12th PROJ PTS 9.3 RB RNK 30th YTD Stats RUYDS 210 REC 4 REYDS 34 TD 0
FPTS/G 7.1 The Steelers' lack of scoring opportunities really crushes the upside for Harris, who admittedly has improved this season and could be on his way to
being a reliable RB2. The Ravens have been pretty good against the run but they can give up some numbers. Harris is a volume-based back who is at best a low-end
starting option given everything that's going on.
</item 1>

<item 2>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.

STARTS: Najee Harris, Kyler Murray, Trey McBride, Marquise Brown, Pat Freiermuth, Diontae Johnson (PPR flex), Jaylen Warren (PPR only), Steelers DST

Najee Harris, Kyler Murray, Trey McBride, Marquise Brown, Pat Freiermuth, Diontae Johnson (PPR flex), Jaylen Warren (PPR only), Steelers DST SITS: Kenny
Pickett, James Conner, Rondale Moore, Cardinals DST

Kenny Pickett, James Conner, Rondale Moore, Cardinals DST SLEEPERS: Greg Dortch (PPR only), George Pickens

Sit Him (Lineup Decisions) Projections powered by Sportsline James Conner RB ARI Arizona • #6

Age: 28 • Experience: 7 yrs.
</item 2>

<item 3>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Patriots

22 Jerome Ford, Browns vs. Jaguars

23 Alexander Mattison, Vikings @ Raiders

24 Najee Harris, Steelers vs. Patriots

25 Ezekiel Elliott, Patriots @ Steelers

26 Gus Edwards, Ravens vs. Rams

27 James Cook, Bills @ Chiefs

28 Zack Moss, Colts @ Bengals

29 Keaton Mitchell, Ravens vs. Rams

30 AJ Dillon, Packers @ Giants

31 Dameon Pierce, Texans @ Jets

32 Aaron Jones, Packers @ Giants

33 Clyde Edwards-Helaire, Chiefs vs. BIlls

34 Jerick McKinnon, Chiefs vs. Bills

35 Roschon Johnson, Bears vs. Lions

36 Tyjae Spears, Titans @ Dolphins

37 Kareem Hunt, Browns vs. Jaguars

38 Tyler Allgeier, Falcons vs. Buccaneers

39 Chuba Hubbard, Panthers @ Saints

40 D'Onta Foreman, Bears vs.
</item 3>

<item 4>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Could this be the weekend that Keaton Mitchell (vs. Rams) gets back into RB2 range for the Ravens? Los Angeles might find itself spying Lamar Jackson and Gus
Edwards so often that the speedy rookie busts a couple of big runs. Mitchell has racked up 330 yards and two TDs since he broke out four games ago, but the
majority of that damage (134 yards and a TD) came in the initial breakout against Seattle. He has still yet to see 12 touches in a game, something we can't
imagine will be the case by the end of the regular season.

We also think Roschon Johnson (vs. Lions) could have a chance to take over lead-back duties in Chicago, as both Khalil Herbert and D'Onta Foreman have battled
injuries and inconsistent touches this season.
</item 4>

<item 5>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
We have the Cowboys back just outside our top five with a juicy matchup against a Washington D that just got lit up by Saquon and the G-men.

Raheem Mostert (@ Jets) should continue to eat for Miami, with rookie De'Von Achane (knee) unfortunately getting hurt almost immediately after returning from IR
(more on that later). Gang Green hasn't been nearly as good against the run as it has been against the pass, as Jets opponents have averaged 139 rushing yards
per game over the past two weeks. 'Fins backup Salvon Ahmed also landed on the IR with a foot injury, so consider stashing Jeff Wilson Jr. as a handcuff.

Rounding out the top 10, the trio of Travis Etienne (@ Texans), Josh Jacobs (vs. Chiefs), and Breece Hall (vs. Dolphins) will be looking to bounce back from
disappointing Week 11 outings on the ground.
</item 5>

<item 6>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Can't say the same against the run as the Redbirds rank in the bottom five in pretty much every single run defense metric except yards after contact per rush
(where they're league average). A running back has scored 17 or more PPR points in four of their past five against the Cardinals, and eight in 12 games have hit
at least 15 PPR points. With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.
</item 6>

<item 7>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Spears contributed 20 yards on the game-tying drive in the fourth quarter and another 27 yards in OT, but Mike Vrabel's squad ultimately lost. In what should be
a high-scoring game against the Super Bowl-hopeful 'Fins, we like Spears to be busy as rookie QB Will Levis's top check-down option -- assuming Henry
(head/neck) is out, of course.

Ezekiel Elliott (@ Steelers) is another potential fill-in who should produce at least flex-worthy stats this week. With Rhamondre Stevenson (ankle) on the
shelf, Elliott could see 20 touches. Similarly, Kenneth Gainwell (@ Cowboys) and Roschon Johnson (vs. Lions) could see big upticks in touches if D'Andre Swift
and D'Onta Foreman can't suit up, but we're not as confident those backs can produce given their matchups.
</item 7>

<item 8>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
</item 8>

<item 9>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
We still like Jaylen Warren better overall, but you can confidently take Najee Harris (vs. Patriots) out of the doghouse for good. Harris has been running with
purpose, and his role in the passing game continues to make him a high-ceiling RB2 in a run-first offense that will be without Kenny Pickett (ankle) this week.
New England might be banged up, but it's still a tough D to get chunk yardage against through the air. Look for Harris to see lots of screens and dump-downs in
a rough-and-tumble Thursday night game in Pittsburgh.

We love Baltimore's Keaton Mitchell (vs. Rams) and think he has the potential to smash Los Angeles at home this week.
</item 9>

<item 10>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Bears

43 Dameon Pierce, Texans vs. Jaguars

44 Royce Freeman, Rams @ Cardinals

45 Jeff Wilson Jr., Dolphins @ Jets

46 Rico Dowdle, Cowboys vs. Commanders

47 Antonio Gibson, Commanders @ Cowboys

48 Miles Sanders, Panthers @ Titans

49 Tyjae Spears, Titans vs. Panthers

50 Clyde Edwards-Helaire, Chiefs @ Raiders

51 Dalvin Cook, Jets vs. Dolphins

52 Zack Moss, Colts vs. Buccaneers

53 Kenneth Gainwell, Eagles vs. Bills

54 Tony Jones Jr., Cardinals vs. Rams

55 Elijah Mitchell, 49ers @ Seahawks

56 Jaleel McLaughlin, Broncos vs. Browns

57 Justice Hill, Ravens @ Chargers

58 Chase Edmonds, Buccaneers @ Colts

59 Boston Scott, Eagles vs.
</item 10>

<item 11>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.
</item 11>

<item 12>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Titans

6 De'Von Achane, Dolphins vs. Titans

7 Breece Hall, Jets vs. Texans

8 Jahmyr Gibbs, Lions @ Bears

9 Alvin Kamara, Saints vs. Panthers

10 Saquon Barkley, Giants vs. Packers

11 Tony Pollard, Cowboys vs. Eagles

12 Bijan Robinson, Falcons vs. Buccaneers

13 Derrick Henry, Titans @ Dolphins

14 Kyren Williams, Rams @ Ravens

15 Josh Jacobs, Raiders vs. VIkings

16 Rachaad White, Buccaneers @ Falcons

17 Joe Mixon, Bengals vs. Colts

18 D'Andre Swift, Eagles @ Cowboys

19 Javonte Williams, Broncos @ Chargers

20 Kenneth Walker III, Seahawks @ 49ers

21 Jaylen Warren, Steelers vs. Patriots

22 Jerome Ford, Browns vs.
</item 12>

<item 13>
Patriots vs. Steelers FanDuel Picks: NFL DFS lineup advice for Week 14 TNF single-game tournaments (Sporting News, 2023-12-07T21:26:00+00:00)
Expect the RB to be fed a ton. Whether that results in a touchdown remains to be seen, but if anyone is scoring a TD on the Pats roster Thursday night, it's
Elliott.

FLEX RB Najee Harris, Steelers ($14,000). Much like the Patriots, this Steelers offense will want to establish the ground game, and that starts with Harris. It
wasn't long ago that the RB was the subject of major criticism for looking slow and ineffective, but that narrative has flipped in a big way. With an average of
15 carries per game in the past five, we can expect at least that and maybe more from a Pittsburgh team that is starting Mitch Trubisky on Thursday night. As an
added bonus, Harris is also the preferred goal-line back, scoring in three of those games. The Patriots' defense doesn't give up much, making Harris our best
bet for success in an old-fashioned defensive struggle between two of the worst offenses the NFL has to offer.
</item 13>

<item 14>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Chiefs

10 Breece Hall, Jets vs. Dolphins

11 Rhamondre Stevenson, Patriots @ Giants

12 Alvin Kamara, Saints @ Falcons

13 Isiah Pacheco, Chiefs @ Raiders

14 Brian Robinson Jr., Commanders @ Cowboys

15 Bijan Robinson, Falcons vs. Saints

16 Jonathan Taylor, Colts vs. Buccaneers

17 James Conner, Cardinals vs. Rams

18 D'Andre Swift, Eagles vs. Bills

19 Joe Mixon, Bengals vs. Steelers

20 Derrick Henry, Titans vs. Panthers

21 Rachaad White, Buccaneers @ Colts

22 Kyren Williams, Rams @ Cardinals

23 Alexander Mattison, Vikings vs. Bears

24 Gus Edwards, Ravens @ Chargers

25 Zach Charbonnet, Seahawks vs. 49ers

26 Javonte Williams, Broncos vs.
</item 14>

<item 15>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.
</item 15>

<item 16>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
3.

We have two sets of teammates in our top 10 this week: Detroit's David Montgomery and Jahmyr Gibbs (@ Bears) and Miami's Raheem Mostert and De'Von Achane (vs.
Titans). Both duos have such solid workloads in high-flying, run-first offenses — and such juicy matchups — that we can't imagine any of these studs finishing
outside the elite eight. DMont and Gibbs combined for 193 yards and two TDs when they faced Chicago in Week 11. Tennessee just surrendered 31 points to Indy, so
we know Miami will be dominating the time of possession and running early and often against Will Levis and company.

Jets fans will be hoping for some Breece Lightning from stud second-year back Breece Hall (vs. Texans), and we think they will be rewarded for their patience.
</item 16>

<item 17>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Eagles

15 Kyren Williams, Rams @ Ravens

16 Joe Mixon, Bengals vs. Colts

17 Javonte Williams, Broncos @ Chargers

18 Derrick Henry, Titans @ Dolphins

19 D'Andre Swift, Eagles @ Cowboys

20 Jaylen Warren, Steelers vs. Patriots

21 Alexander Mattison, Vikings @ Raiders

22 Jerome Ford, Browns vs. Jaguars

23 Kenneth Walker III, Seahawks

24 Najee Harris, Steelers vs. Patriots

25 James Cook, Bills @ Chiefs

26 Zack Moss, Colts @ Bengals

27 Keaton Mitchell, Ravens vs. Rams

28 AJ Dillon, Packers @ Giants

29 Ezekiel Elliott, Patriots @ Steelers

30 Aaron Jones, Packers @ Giants

31 Jerick McKinnon, Chiefs @ Bills

32 Tyjae Spears, Titans @ Dolphins

33 Gus Edwards, Ravens vs. Rams

34 Dameon Pierce, Texans @ Jets

35 Roschon Johnson, Bears vs.
</item 17>

<item 18>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
49ers

26 Javonte Williams, Broncos vs. Browns

27 AJ Dillon, Packers @ Lions

28 Jerome Ford, Browns @ Broncos

29 Devin Singletary, Texans vs. Jaguars

30 Kareem Hunt, Browns @ Broncos

31 James Cook, Bills @ Eagles

32 Jaylen Warren, Steelers @ Bengals

33 Najee Harris, Steelers @ Bengals

34 Khalil Herbert, Bears @ Vikings

35 Tyler Allgeier, Falcons vs. Saints

36 Chuba Hubbard, Panthers @ Titans

37 Ezekiel Elliott, Patriots @ Giants

38 Latavius Murray, Bills @ Eagles

39 Roschon Johnson, Bears @ Vikings

40 Joshua Kelley, Chargers vs. Ravens

41 Keaton Mitchell, Ravens @ Chargers

42 Ty Chandler, Vikings vs. Bears

43 Dameon Pierce, Texans vs.
</item 18>

<item 19>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He's still a decent start ahead of Najee Harris, Khalil Herbert and Jahmyr Gibbs but I'd rather chance my lineup with someone who has upside like Jaleel
McLaughlin (if Javonte Williams sits out) or Alexander Mattison.

FILM: The numbers don't quite match the talent -- Stevenson is a capable runner with great vision, balance and power to go with good speed. He struggled on
lateral runs last week and his line failed him plenty on north-south runs. Game script kept him from finding a lot of work last week (one fourth-quarter touch).

THIS SEASON: Stevenson has at least 16 touches in every game, which is good. Unfortunately he has yet to total 100 yards and has been under 70 yards three
times, which is definitely the opposite of good.
</item 19>

<item 20>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We'll talk more about Derrick Henry (concussion) in the injury section, but we can say with confidence here and now that Tyjae Spears (@ Dolphins) should be the
hottest waiver-wire pickup of Week 14. Tennessee is a run-first offense and King Henry might be sidelined with a head injury, so Spears could very well pick up
where he left off against Indy last weekend. The young back averaged 4.7 yards per carry (just shy of his 5.1 YPC entering the week) en route to 88 yards.

Another backup getting a boost this week is the Patriots' Ezekiel Elliott (@ Steelers), who will assume lead-back duties after Rhamondre Stevenson went down
with a pretty bad ankle injury in Week 13. Zeke might put in a vintage performance considering James Conner just went back to Pittsburgh and hung 105 yards and
two TDs on his old squad.

Could this be the weekend that Keaton Mitchell (vs.
</item 20>
=== END ITEM id=114bbc9a0b6ced0f ===
