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

=== ITEM id=c088cee25cb741c8 ===
Gold sentences and facts:
Sentence 0: The process also loops in the European Commission to help facilitate dialogue, assess issues and bring pressure to bear on unfair practices.
  Fact 0.0: The process also involves the European Commission.
  Fact 0.1: The European Commission helps facilitate dialogue.
  Fact 0.2: The European Commission helps assess issues.
  Fact 0.3: The European Commission helps bring pressure to bear on unfair practices.
Sentence 1: We continue to work through the process with the European Commission and are focused on addressing its questions and any identified concerns at this stage.
  Fact 1.0: The speaker continues to work through the process with the European Commission.
  Fact 1.1: The speaker is focused on addressing the European Commission's questions.
  Fact 1.2: The speaker is focused on addressing any identified concerns at this stage.
Sentence 2: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 2.0: The earlier actions were focused on concerns about the spread of illegal content.
  Fact 2.1: The earlier actions were focused on concerns about disinformation.
  Fact 2.2: The illegal content and disinformation are related to the Israel-Hamas war.

Retrieved items:
<item 1>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Time will tell how this one plays out but he may find the dead blue bird he’s chosen to hang around his neck, when he purposefully ripped up the standard
content moderation rulebook and opted to channel online chaos, starting to feel like it’s dragging him down.
</item 1>

<item 2>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
(And on that front, Norway’s DPA, which has a local ban order on Meta’s tracking ads, told us it’s concerned about the subscription plan, saying it has doubts
as to whether it would constitute valid consent.)

As well as needing to comply with the GDPR, which delineates the qualities necessary for consent to be legal (such as that it is specific, informed and freely
given), Meta is now also subject to the pan-EU Digital Services Act (DSA) — which also sets conditions on larger platforms when it comes to tracking and
profiling people for ads. So it won’t just be data protection authorities making a call on whether Meta’s subscription or tracking offer flies; the European
Commission is responsible for oversight of very large online platforms’ DSA compliance.

Meta has also been designated as a so-called gatekeeper, under the DSA’s sister regulation, the Digital Markets Act (DMA) — which also puts some limits on use
of people’s data for ads. The Commission is the sole enforcer of the DMA.
</item 2>

<item 3>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
So — tl;dr — if EU officials have used X’s ad-targeting tools to break the bloc’s own digital rulebook it’s the very definition of an awkward situation.

The existence of the Commission’s microtargeted ad campaign seeking to drum up support for its proposed CSAM-scanning law was spotted last month by
technologist, Danny Mekić. An article with his findings ran in Dutch newspaper, De Volkskrant, earlier this month.

Using public ad transparency tools the DSA requires VLOPs to provide, Mekić found the Commission had run a paid advertising campaign on X, targeting users in
the Netherlands, Sweden, Belgium, Finland, Slovenia, Portugal and the Czech Republic — countries that were not supportive of Johansson’s CSAM-scanning proposal
according to leaked minutes from a September 14 meeting of the European Council, a co-legislative body that’s involved (along with MEPs) in determining the
final shape of the CSAM law.
</item 3>

<item 4>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
It replied with an automated email reading: “Busy now, please check back later.”

Update: X’s global government affairs account has posted a thread responding to the EU’s criticism in which it claims the data shows other social networking
services saw greater changes in subscriber growth and total growth for pro-Kremlin accounts, and for engagement and volume; it also writes that: “More than 700
unique Community Notes have appeared on posts related to the Ukraine conflict. Through Community Notes’ new ‘Notes on Media’ feature, which automatically
matches notes on an image or video to other posts that contain matching media, these notes appeared on over 1,200 additional posts. These notes are in numerous
languages and are written specifically for local audiences.”

Back in April, X/Twitter was named as one of 19 larger platforms that face the strictest level of regulation under the EU’s DSA.
</item 4>

<item 5>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
“We’re continuing to explore how to provide teens with a useful and responsible ad experience given this evolving regulatory landscape,” it writes on that.
</item 5>

<item 6>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
But at the time of writing, Musk had engaged in a bit of a dance of his own with the news of the EU’s warning by responding to a critical tweet posted to X by
journalist Glenn Greenwald, who attacked the EU’s new “censorship law,” as he dubbed the DSA — which he claimed would be used to “punish X.”

In reply to Greenwald, Musk avoided expressing the same trenchant criticism of the EU’s approach but invited yet more relativism — writing: “Let the public hear
exactly what this disinformation consists of and decide for themselves.”

He then went on to sew doubt that disinformation is something that can be independently arbitrated at all, implying fact-checking is just a convenient exercise
for targeting different opinions — so essentially echoing Greenwald’s position — by claiming that “many times” the “official fact-checker” has been found making
false statements, before adding a rhetorically empty offer that “Maybe this is the case here, maybe not” for good, plausible deniability measure.
</item 6>

<item 7>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.
</item 7>

<item 8>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
Even for a user with just one account (on either Facebook or Instagram) the cost for protecting their privacy from Meta’s tracking and profiling would be almost
€120 per year (for web use) or just over €155 (on mobile).

As we reported earlier this month, Meta is relying on a line in a ruling handed down by the bloc’s top court, the CJEU, earlier this year — where the judges
allowed the possibility — caveated with “if necessary” — of an (another caveat) “appropriate fee” being charged for an equivalent alternative service (i.e. that
lacks tracking and profiling). So the legal fight against Meta’s continued tracking and profiling of users will hinge on what’s necessary and appropriate in
this context.

noyb, the European privacy rights group which has driven much of the strategic litigation against Meta’s tracking and profiling, has already — since 2021 — been
challenging similar ‘pay or okay’ practices by news publishers by filing a series of complaints with data protection authorities.
</item 8>

<item 9>
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
</item 9>

<item 10>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Days after a privacy complaint was lodged against Meta in the European Union over its latest controversial shift of legal basis claimed for processing people’s
data for ads, consumer groups across the region are filing their own complaints about what the tracking giant is up to.

A coalition of almost 20 consumer protection organizations is united in the view that Meta’s switch to railroading users into agreeing to being tracked and
profiled so it can keep profiting from microtargeting them is “unfair” and “illegal” — breaching EU consumer protection law “on several counts”.

Starting this month, EU users of Meta’s social networks, Facebook and Instagram, are being offered the ‘choice’ of agreeing to being tracked and profiled by the
behavioral ads business in order to continue/get free access to its products — or else they must pay it a monthly subscription (of at least €9.99pm) for an
ad-free version of its mainstream social networks.
</item 10>

<item 11>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” he warns. “We have, from qualified sources, reports about potentially illegal content circulating on your service despite flags from relevant
authorities.”

He also takes issue with a change X made last night to its public interest policy, under which it judges newsworthiness (i.e., for deciding whether posts that
infringe its rules may nonetheless remain on the site) — but which Breton says have left “many European users uncertain” (i.e., about how X is applying its own
rules).

Again, this is a problem because the DSA requires platforms to be clear and transparent about their rules and how they apply them. “This is particularly
relevant when it comes to violent and terrorist content that appears to circulate on your platform,” Breton continues in another pointed warning.
</item 11>

<item 12>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Commercial influence

While many of the questions directed at her over the 1.5-hour long hearing focused on the controversy that’s sprung up around the ad campaign, parliamentarians
also pressed the commissioner on a number of other issues — including concerns about the extent of commercial lobbying around the CSAM-scanning proposal.

This has been a topic of intense interest, especially following a report by investigative journalists published last month by BalkanInsight which looked at
close contacts between Johansson’s department and companies with CSAM-scanning and other child safety tools to sell.

One of the journalists involved in that investigation, Apostolis Fotiadis, had also been invited by the committee to participate in the exchange of views — and
he took the opportunity to defend their reporting from direct public attacks by Johansson.
</item 12>

<item 13>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
This law created very specific restrictions around young folks online, limiting data collection and requiring things like parental consent for various actions,
but a lot of tech companies seem to consider it more suggestion than requirement.

You know it is bad news for the company when they request pages and pages of redactions:

This recently happened with Amazon as well, and it turned out they were trying to hide the existence of a price-hiking algorithm that skimmed billions from
consumers. But it’s much worse when you’re redacting COPPA complaints.

“We’re very bullish and confident in our COPPA allegations. Meta is knowingly taking steps that harm children, and lying about it,” AG Bonta told TechCrunch in
an interview. “In the unredacted complaint we see that Meta knows that its social media platforms are used by millions of kids under 13, and they unlawfully
collect their personal info. It shows that common practice where Meta says one thing in its public-facing comments to Congress and other regulators, while
internally it says something else.”
</item 13>

<item 14>
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
</item 14>

<item 15>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
In the event MEPs got some bare bones detail.

Asked for a list of these contacts, the commissioner responded that she’d met with Google six times; Microsoft, Meta and TikTok three times each; twice with
Twitter (X); and once apiece with Apple and Amazon. She also said she’d met with the child safety organizations Thorn (twice) and Brave Movement (twice); and
with Tech Alliance and ICANN once apiece.

In wider responses related to concerns about how much commercial interest had influenced the Commission, Johansson highlighted her decision for the
CSAM-scanning proposal to be “technology neutral” — meaning the draft regulation does not support any specific tech solution — with the suggestion being EU
lawmakers had resisted lobbying by companies for a law that would explicitly favor their existing tech tools.

She also denied that only Thorn and Microsoft have technology “that is necessary for the scanning” — claiming that’s “absolutely not true”.

“There are no specific technologies mentioned [in the proposal] and I think this is an important part.
</item 15>

<item 16>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In this case, failure to reply by the deadline could lead to the imposition of period penalties,” the EU added.

Earlier this year, Musk pulled out of the EU’s Code of Practice on online disinformation. In a response to X leaving the voluntary Code, Breton warned:
“Obligations remain. You can run but you can’t hide.”

In recent days, TikTok and Meta have also been warned by the EU over disinformation related to the Israel-Hamas war. But back in September, a study commissioned
via the EU’s Code of Practice suggested X is the worst of the major platforms when it comes to spreading disinformation.

This report was substantially revised to clarify a number of aspects, including — most saliently — that the EU has sent a formal request for information to X at
this point. This could prefigure a formal announcement of an investigation but is not, yet, that formal technical step.
</item 16>

<item 17>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
At the time the EU’s then competition chief, Margrethe Vestager, trumpeted the outcome as setting what she dubbed “new rules for how Amazon operates its
business in Europe” — arguing the settlement meant the ecommerce giant could “no longer abuse its dual role” as marketplace operator and own brand retailer; and
further claiming the outcome would ensure “competing independent retailers and carriers as well as consumers will benefit from these changes opening up new
opportunities and choice.”

Given the commitments the EU already extracted from Amazon after those earlier investigations it’s interesting to see the bloc remains concerned over levers the
ecommerce behemoth still has its disposal to potentially influence competitive outcomes on its marketplace vis-à-vis iRobot vs rival robot vacs.

Since then Amazon has also been named a gatekeeper under the EU’s shiny new ex ante competition regulation, the Digital Markets Act — which lists its
marketplace as a core platform service. This designation means Amazon must comply with a series of up-front obligations, which include restrictions on self
preferencing.
</item 17>

<item 18>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
Meta is to offer an ad-free subscription version of Facebook and Instagram in the European Union, EEA (European Economic Area) and Switzerland, confirming the
core of a report in the WSJ earlier this month. The new ad-free subscription will be available from next month, per a Meta blog post.

The move follows years of privacy litigation, enforcements and court rulings in the EU — which have culminated in a situation where Meta can no longer claim a
contractual right (nor legitimate interest) to track and profile users for ad targeting. (Although, at the time of writing, it is still doing the latter —
meaning it is technically operating without a proper legal basis. But this summer Meta announced an intention to switch to consent.)

Under regional data protection law the only available basis left for Meta’s tracking and profiling ad business is to obtain freely given consent from users.
</item 18>

<item 19>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
He’s also ripped up a bunch of legacy content moderation policies and slashed in-house enforcement teams while promoting a decentralized, crowdsourced
alternative (rebranded as Community Notes), which essentially outsources responsibility for dealing with tricky issues like disinformation to users in what
looks suspiciously like another gambit to eke out extra engagement and farm confusion by applying a philosophy of extreme relativism so culture warriors are
encouraged to keep forever fighting for their own “truth” in the comments.

Oh and he also pulled X out of the EU’s Code of Practice on Disinformation earlier this year in a very clear thumb of the nose to EU regulators.

Urgent letter to Musk

In an “urgent” letter to Musk today, which the EU’s internal market commissioner Thierry Breton also shared on X, the bloc has sent the strongest signal yet it
believes Musk’s platform is in breach of the DSA — although this is by no means Musk’s first warning.
</item 19>

<item 20>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
So when we say facilitate, it’s obvious that the EU officials discuss what kind of experts will be available from this particular stakeholder to attend these
meetings and to present the point of view, which seems to be a privileged access,” he explained.

“Also in the same email thread there’s mention of EU officials being allocated to specifically attend the cooperation between the stakeholder and DG-Home on the
proposal, which to our understanding is something that goes way beyond the level of consultations or exchange of views or exchange of opinions on the proposal
and speaks directly to cooperation.”

The committee took the opportunity to press Johansson about her contacts with companies and other lobbyists during the drafting of the CSAM-proposal, with MEPs
saying they want clear answers to the allegations of commercial interest and heavy lobbying when the Commission was setting up and drafting the proposal.

In the event MEPs got some bare bones detail.
</item 20>

<item 21>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The seller went on to describe the experience as a “race to the bottom.” “It's long overdue," another commenter wrote. "When they close me down, I'm applying
for a job with the FTC.” Still, others commented that the FTC’s complaint is misguided. “Selling on Amazon is a life-changing opportunity and the amount of
sellers that throw stones at the platform is astounding,” one user wrote.

Seller skepticism

Even sellers who may be sympathetic to the idea of regulating Amazon have concerns, specifically that the FTC's highlighted issues aren't necessarily ones that
would make the seller and consumer experience better. Scott Needham, who sells on Amazon and runs a product-finder tool for other Amazon sellers, said he was
“surprised by some of the points that the FTC selected.”

watch now

“I have over the years been very critical of Amazon,” Needham told CNBC. “I’ve lost a lot of sleep because of some of the things that they have done.
</item 21>

<item 22>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
In a blog post ahead of today’s hearing — which deploys a crisis-PR-esque headline claim of “setting the record straight” — she criticized the article as “a
series of insinuations looking for a home”; claiming it paired an outline of “a selection of meetings I had, of events I attended, or conferences I addressed”
with “a conspiratorial tone” in an attempt “to create the impression of financial influence where there is none”.

Fotiadis was asked by the LIBE committee about the accusation that the journalists had, essentially, been spreading disinformation — and specifically whether he
believed Johansson and the Commission’s response to it amounted to a restriction on media freedom. He responded by saying he did not think that was the case.
But went on to express surprise at how the Commission had reacted to the scrutiny — to its instinct to deploy “spin-doctor” tactics to try to discredit the
article, rather than engaging with the substance of the concerns being raised.
</item 22>

<item 23>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
The process also loops in the European Commission to help facilitate dialogue, assess issues and bring pressure to bear on unfair practices.

The CPC alert and mobilization process can be quicker than GDPR enforcement when it comes to forcing changes to unfair behaviors. Although it still typically
takes months, plural, for the network to coordinate and arrive at a position to press on a trader they believe is infringing the law.

The network also can’t impose fines itself. But if issues aren’t resolved through the dialogues and commitments process it shoots for, national consumer
protection authorities can still pursue enforcement at a local level. So if, at the end of the day, these consumer advocacy groups aren’t happy with whatever
the process of pressing Meta for changes will have achieved they can still press complaints to national authorities to urge them to take enforcement action (and
those CPAs have the ability to impose penalties of up to 4% of global annual turnover).
</item 23>

<item 24>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
In this way, Meta goes through great lengths to avoid meaningfully complying with COPPA, looking for loopholes to excuse its knowledge of users under the age of
13 and maintain their presence on the Platform.

The other complaints in the lengthy lawsuit have softer edges, such as the argument that use of the platforms contributes to poor body image and that Meta has
failed to take appropriate measures. That’s arguably not as actionable. But the COPPA stuff is far more cut and dry.

“We have evidence that parents are sending notes to them about their kids being on their platform, and they’re not getting any action. I mean, what more should
you need? It shouldn’t even have to get to that point,” Bonta said.

“These social media platforms can do anything they want,” he continued. “They can be operated by a different algorithm, they can have plastic surgery filters or
not have them, they can give you alerts in the middle of the night or during school, or not.
</item 24>

<item 25>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
the so-called Blue checks”

“If proven, these failures would constitute infringements of Articles 34(1), 34(2) and 35(1), 16(5) and 16(6), 25(1), 39 and 40(12) of the DSA,” the EU added —
saying it will now carry out an in-depth investigation “as a matter of priority”.

Today we open formal infringement proceedings against @X : ⚠️ Suspected breach of obligations to counter #IllegalContent and #Disinformation ⚠️ Suspected breach
of #Transparency obligations ⚠️ Suspected #DeceptiveDesign of user interface#DSA pic.twitter.com/NxKIif603k — Thierry Breton (@ThierryBreton) December 18, 2023

The Commission is responsible for enforcing the DSA on larger platforms such as X — which was designated a very large online platform (VLOP) under the DSA back
in April.
</item 25>

<item 26>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Some DPAs appear to have been sympathetic to local newspapers’ use of cookie walls, seeing it as a way to support the production of journalism. However when it
comes to Meta, that argument evaporates as it’s definitely not in the journalism business. Moreover the adtech giant doesn’t even need to produce content to
pump around its social networks; it gets all that filler for free from the self-same users it’s now demanding pay a fee if they want to use its services without
being tracked and profiled for behavioral advertising. Which, well, makes Meta’s ‘pay or okay’ model feel like even more of a rip off.

Back in April, a decision by Austria’s DPA on a noyb complaint about cookie paywalls said users must have the ability to say yes or no to specific data
operations — meaning blanket consent is not an option.
</item 26>

<item 27>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
But in the release, the AI Alliance stresses that its work — whatever form it ultimately takes — is intended to be complementary and additive rather than
needlessly duplicative.

“[M]ore collaboration and information sharing will help the community innovate faster and more inclusively, and identify specific risks and mitigate those risks
before putting a product into the world,” the release reads. “This stands in contrast to a vision that aims to relegate AI innovation and value creation to a
small number of companies with a closed, proprietary vision for the AI industry.”

Key subtext

That jab at the end says a lot about Meta’s ulterior motives, here.

Google, OpenAI and Microsoft, a close OpenAI partner and investor, have been among the chief critics of Meta’s open source AI approach, arguing that it’s
potentially dangerous and disinformation-encouraging. (Unsurprisingly, none are members of the AI Alliance despite being longtime members of the Partnership on
AI.)
</item 27>

<item 28>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
In recent times, a raft of complaints to the CPC about TikTok led — just last year — to the video sharing social network pledging to improve user reporting and
disclosure requirements around ads/sponsored content; and to boost transparency around its digital coins and virtual gifts. Although BEUC was not ecstatic about
the outcome, saying significant concerns remained unaddressed.

Still, the CPC network may be able to extract some ‘quick win’ concessions from Meta — such as requiring it to amend how it presents the available ‘Hobson’s
choice’ to users. Meta could also potentially face pressure to lower the subscription fee to make it more more affordable for users to deny tracking. (Just
spitballing here but imagine if it were offering a choice of tracking ads vs paying €1 a year not to be tracked which wouldn’t look so evidently self-serving.)
</item 28>

<item 29>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
A rule of law clash between Musk and the EU has looked increasingly inevitable in recent months — and, indeed, has been predicted by some industry watchers
since rumors of the erratic billionaire’s plan to take over Twitter emerged last year.

Reminder: Penalties for confirmed breaches of the DSA can be as high as 6% of global annual turnover. The bloc also has powers, in extremis, which could result
in access to X being shut off in the region if the platform repeatedly fails to correct course. So if Musk fails to satisfy EU regulators, he could face serious
consequences for what remains a highly indebted business.

In his letter to Musk, Breton writes that the EU has “indications” that X is being used to disseminate illegal content and disinformation in the EU” following
Saturday’s attacks. He then goes on to remind the company of the DSA’s “very precise obligations” vis-à-vis content moderation.
</item 29>

<item 30>
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
</item 30>

<item 31>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
We also clarified the scale of financial penalties possible under the DSA if a breach is confirmed; and added details of the regulation’s crisis response
mechanism. We also fixed an error in the original report which referred to Interpol, rather than Europol.
</item 31>

<item 32>
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
</item 32>

<item 33>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.
</item 33>

<item 34>
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
</item 34>

<item 35>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 35>

<item 36>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
European Union regulators have sent Amazon a formal statement of objections setting out competition concerns over its proposed acquisition of robot vacuum
maker, iRobot.

The step does not confirm the EU will seek to block the deal but it shows antitrust regulators do remain concerned. It could therefore push Amazon to offer
remedies to EU authorities to settle the investigation and avoid the risk of the acquisition being blocked.

Amazon has already lowered its acquisition price for iRobot — which was originally an all-cash offer of $1.7 billion back in August 2022 — by 15% owing to
increased debt taken on by its target while regulatory snags delayed closing the deal at the original purchase price.

The bloc has been closely scrutinizing the Amazon-iRobot deal since July when the EU announced its in-depth probe. The Commission said at the time it was
worried the transaction would allow Amazon to restrict competition in the market for robot vacuum cleaners (RVCs) and strengthen its position as online
marketplace provider in a number of ways.
</item 36>

<item 37>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
The law puts an obligation on so-called very large online platforms (VLOPs) to assess societal risks attached to their use of algorithms — and put in place
“reasonable, proportionate and effective mitigation measures” for identified risks, with their reporting and mitigation plans subject to independent audit and
oversight by the Commission.

We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.

Update: A Commission spokesperson told us:

While the Code is voluntary, compliance with the DSA is not. The DSA is now legally binding on X/Twitter. This includes in particular the obligation of Very
Large Online Platforms, such as X/Twitter to mitigate the risks their platforms pose for the spread of disinformation. The TrustLab study to measure two
Structural Indicators covers signatories of the Code at the time of the launch of the exercise, which at the time included X/Twitter.
</item 37>

<item 38>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“I remind you that following the opening of a potential investigation and a finding of non-compliance, penalties can be imposed,” Breton adds, concluding the
letter.

We’ve asked the Commission to confirm whether it has opened an investigation into X’s DSA compliance over concerns it’s raised in the letter. Perhaps it’s going
to wait a day to see what his response is before taking that next dance step.

Musk’s extremely iterative — or just plain erratic/arbitrary — management style is so far from the responsive and responsible qualities the EU’s rulebook
demands of digital leaders it’s hard to see how this clash can end well for either side.

We contacted X for a response to the EU’s warnings about its DSA compliance but at press time the company had not responded — beyond firing back its usual
automated reply, which reads: “Busy now, please check back later.”
</item 38>

<item 39>
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
</item 39>

<item 40>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
Ofcom’s guidelines, for example, specify that users should be able to easily block users — but Musk has publicly stated his intentions to remove X’s block
feature. He’s clashed with the EU over similar rules and reportedly even considered pulling out of the European market to avoid them. Whitehead declined to
comment when I asked whether X had been cooperative in talks with Ofcom but said the regulator had been “broadly encouraged” by the response from tech firms
generally.

“We think they represent best practice of what’s out there, but it’s not necessarily applied across the board.”

Ofcom’s regulations also cover how sites should deal with other illegal harms like content that encourages or assists suicide or serious self-harm, harassment,
revenge porn and other sexual exploitation, and the supply of drugs and firearms. Search services should provide “crisis prevention information” when users
enter suicide-related queries, for example, and when companies update their recommendation algorithms, they should conduct risk assessments to check that
they’re not going to amplify illegal content.
</item 40>

<item 41>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
The EU has reached this pass of laying out formal concerns after conducting what it describes as a “wide-ranging investigation to understand the market and the
potential impact of the deal”, which it says included analysing internal documents provided by Amazon and iRobot and gathering views from market participants
such as suppliers of RVCs and other smart home devices, as well as from providers of online sales channels.

It also says it closely cooperated with other competition authorities during both the initial investigation and the in-depth investigation, adding that it will
continue to do so during the remainder of the in-depth investigation. Although the U.K.’s competition authority already cleared the deal back in June.

Amazon was contacted for a response to the EU’s statement of objections. A company spokesperson sent us this statement in which it claims iRobot faces “intense
competition” from other RVC suppliers:

We continue to work through the process with the European Commission and are focused on addressing its questions and any identified concerns at this stage.
</item 41>

<item 42>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
So there’s no specific technology that’s been favoured in this proposal,” she also told the committee, adding: “So many technologies are being developed all the
time — while we are speaking — and they will continue to develop. So I think it’s important that the legislation has to be technology neutral.”

Earlier this week a seminar organized by the European Data Protection Supervisor (EDPS), an advisory body to the Commission on data protection law, heard from
more than 20 speakers across civil society, academia and industry expressing deep misgivings about the Commission’s approach — including a warning from the EDPS
himself that the EU could be at a tipping point for freedom and democracy if it does not turn back from the plan to do non-targeted scanning of private
messages.

Johansson had been invited to participate in the seminar but declined to attend. She didn’t offer a direct response to the EDPS’ concerns today but she did
counter a number of arguments heard at the session earlier in the week — including refuting the suggestion that her proposal amounts to mass surveillance.
</item 42>

<item 43>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Given the market power of Meta’s Facebook and Instagram services in the EU and the very strong network effects of social media platforms (since all your friends
are on Facebook and Instagram), consumers do not have a real choice because if they quit the services they would lose all their contacts and interactions built
over the years. The very high subscription fee for ‘ad-free’ services is also a deterrent for consumers, which means consumers do not have a real choice.

“The company’s approach also raises concerns regarding the GDPR,” Pachl further noted. And a spokesman for BEUC told us it might, at a later stage, file a
complaint about Meta’s data protection compliance with the relevant privacy authority, once it has completed its own assessment of the issues. Although he
emphasized it’s too early to say whether or not it will take that step.

Meta’s lead data protection authority in the EU, Ireland’s Data Protection Commission (DPC), has, for several months, been assessing its pay or consent offer.
</item 43>

<item 44>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
On the surface, Microsoft is currently looking like the biggest winner amidst the OpenAI turmoil, as it was already apparently looking to reduce its reliance on
OpenAI even though it remains once of its major shareholders. But Facebook’s parent Meta could also stand to benefit, as businesses pursue multi-modal
strategies or models with a more “open” ethos embedded.

“Open source today offers a wide variety of models for companies to essentially diversify,” Ceze added. “By doing so, these startups can quickly pivot and
minimize risk. There is also a major upside — many of these models already outperform the likes of OpenAI’s in terms [of] price-performance and speed.”

A leaked internal memo from Google earlier this year seemed to express fears that despite the huge advances made by proprietary LLM models from the likes of
OpenAI, open source AI would ultimately trump them all. “We have no moat, and neither does OpenAI,” the document noted.
</item 44>

<item 45>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
Ofcom recommends these platforms implement policies like not allowing strangers to send direct messages to children, using hash matching to detect and remove
CSAM, maintaining content and search moderation teams, and offering ways for users to report harmful content.

Large tech platforms already follow many of these practices, but Ofcom hopes to see them implemented more consistently. “We think they represent best practice
of what’s out there, but it’s not necessarily applied across the board,” Whitehead says. “Some firms are applying it sporadically but not necessarily
systematically, and so we think there is a great benefit for a more wholesale, widespread adoption.”

There’s also one big outlier: the platform known as X (formerly Twitter). The UK’s efforts with the legislation long predate Elon Musk’s acquisition of Twitter,
but it was passed as he fired large swaths of its trust and safety teams and presided over a loosening of moderation standards, which could put X at odds with
regulators.
</item 45>

<item 46>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
The European Union’s home affairs commissioner, Ylva Johansson, has confirmed the Commission is investigating whether or not it broke recently updated digital
governance rules when her department ran a microtargeted political ad campaign aiming to drive support for a controversial child sexual abuse material
(CSAM)-scanning proposal she’s spearheading.

But at a committee hearing in the European Parliament today she deflected MEPs’ enquiries for her to give more details about the ad campaign.

The governance regulation concerned is the Digital Services Act (DSA), which includes provisions relating to online advertising — including a prohibition on the
use of sensitive personal data, such as political opinions, for targeting ads. While the ads in question ran on X (formerly Twitter) — which is already expected
to be compliant with the DSA, having been designated by the Commission as a so-called Very Large Online Platform (VLOP) back in April.

The Commission itself, meanwhile, not only proposed this pan-EU law but is responsible for oversight of VLOPs’ DSA compliance.
</item 46>

<item 47>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
In the latter case its social networks, Facebook and Instagram, were designated as very large online platforms (VLOPs) earlier this year. And, since late
August, they’ve been expected to be compliant with that digital rulebook.

Both pan-EU laws put restrictions on the use of personal data for advertising — explicitly requiring consent is obtained from users for such a purpose; and that
consent must be as easy to withhold as to affirm. So one issue the Commission — which is the sole enforcer of the DSA on VLOPs — might weigh in in the coming
months on is whether clicking accept vs digging out a credit card to pay a monthly charge are equivalently easy.

The regulation also contains provisions which are intended to combat unfair/deceptive design, such as targeting choice interfaces that make it “more difficult
or time-consuming” to pick one option over another.
</item 47>

<item 48>
Twitch’s money guy talks about the revenue split controversy and its monetization long game (TechCrunch, 2023-10-26T22:28:10+00:00)
I have to wonder if those decisions to change these policies were in response to competition from other platforms.

I would say it this way: I think they are an acknowledgement, and maybe a closer connectivity to the community via Dan specifically. I credit Dan in his short
time as CEO to really make some big, hard decisions. Both the launch of Partner Plus, as well as simulcasting, the embed stuff we talked about in the keynote —
these are three things that I think are very much an acknowledgement of the feedback from the community. We’ve made that change, so I wouldn’t call it out of
competitive pressures. I would call it out of focus on the community.
</item 48>

<item 49>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
Asking Musk to be “effective” against disinformation is a bit like asking the sea to stop moving. But, well, this is how the regulatory dance must go (and after
the dance comes the denouement — which, if a DSA breach is confirmed, means enforcement. And actual penalties might be a bit harder for Musk to troll).

In the meantime, the EU has asked Musk to make contact with relevant law enforcement authorities and Europol — and “ensure that you respond promptly to their
requests.” Breton also flags some unspecified additional DSA compliance issues he says his team will be contacting Musk’s about “shortly” — “with a specific
request.”. (We’ve asked the EU what else it’s concerned about on X and will update this report with any response.)

The bloc has given Musk a deadline of 24 hours to respond to its asks at this point — stipulating his answer will be added to its assessment file on X’s
compliance with the DSA.
</item 49>

<item 50>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
And I think it’s a good practice to do so. Because we are we are taking stance and we should defend our stance,” she told the committee.

A number of MEPs pushed back — including by pointing out that there are more appropriate channels for the Commission to engage directly and transparency with
co-legislators than opaque behavioral ad targeting on platforms like Twitter/X.

“One principle of democracy is that we have procedures because the end doesn’t justify the means,” opined MEP Sophie in ‘t Veld. “And European Commission has
the right to be very attached to its legislative proposals but there are privileged channels for the European Commission to communicate with the two legislators
and others — not an ad campaign on Twitter.”

Despite a lot of pushback, the committee was unable to extract any other lines from commissioner on the ad campaign. But at the end of the session she did agree
to respond to it in writing with some missing answers “as soon as possible” (albeit, avoiding agreeing to do so by the end of the week, as one MEP had asked).
</item 50>

<item 51>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Of course we always have to comply with the regulation. There’s no question about that.”

The LIBE committee repeatedly pressed Johansson to provide detailed about the microtargeted ad campaign — but she declined to do so, saying she did not have any
information about it and that it was for her “service”, who she suggested had been responsible for the campaign, to answer. So there was no explanation about
why, for instance, Christians had been explicitly excluded from the Commission’s microtargeting.

She also avoided giving a direct response to accusations by MEPs that the use of political microtargeting by the Commission was anti-democratic — opting instead
to mount a general defence of its right to promote its proposals. She also listed a number of other departments within the Commission she said had previously
used ads to promote separate legislative proposals.

“I think that the commission should defend and explain and promote our proposals. We do that and we have done that. And I think it’s a good practice to do so.
</item 51>

<item 52>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
In a press release earlier this month after the WSJ reported Meta plans to charge users for their privacy, noyb’s founder and honorary chairman, Max Schrems,
wrote: “The CJEU said that the alternative to ads must be ‘necessary’ and the fee must be ‘appropriate’. I don’t think €160 a year is what they had in mind.
These six words are also an ‘obiter dictum’, a non-binding element that went beyond the core case before the CJEU. For Meta this is not the most stable case law
and we will clearly fight against such an approach.”

Contacted for a response to the development, Meta’s lead regulator in the EU for the General Data Protection Regulation (GDPR), Ireland’s Data Protection
Commission (DPC) sent us a statement.
</item 52>

<item 53>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Given her reliance on pointing to the existence of a wider EU legal framework doing the heavy lifting and protecting Europeans’ fundamental rights as a strategy
to assuage critics, and given she’s also invoking respect for the rule of law as a buttress against the risk of content-scanning mission creep, it’s doubly
relevant that the Commission now finds itself in a bind — forced to investigate whether its own officials ignored legal requirements in a bid to covertly sweep
past critics.
</item 53>

<item 54>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 54>

<item 55>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
iRobot, which faces intense competition from other vacuum cleaner suppliers, offers practical and inventive products. We believe Amazon can offer a company like
iRobot the resources to accelerate innovation and invest in critical features while lowering prices for consumers.

Late last year the EU settled with Amazon over two earlier competition probes — one delving into antitrust concerns about Amazon’s use of merchant data to
bolster its own retail business; and a second one looking at how it operated the “Buy Box” and its Prime loyalty program.

In those cases Amazon agreed a series of commitments to end the probes — including saying it would stop using non-public data from its marketplace sellers and
boost transparency for merchants; and pledging to treat all sellers equally, regardless of whether they pay for its logistics services.
</item 55>

<item 56>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
“Meta notified the DPC on July 27 of its intention to implement an alternative, consent-based model in which users would be given a choice between ads-funded
versions of its platforms, and subscription versions in which, in return for payment of monthly fee, it is said that users will not receive targeted
advertising,” the Irish regulator wrote.

“While February 2024 was initially identified by Meta as the earliest date by which its consent model would be operational, it agreed to bring forward that date
to November 2023 at the direction of the DPC, the DPC having been concerned to ensure that changes would be implemented to the platforms as soon as practicable
in light of earlier findings to the effect that Meta had failed to demonstrate its entitlement to rely on the legal bases on which it was then relying when
processing users’ data for behavioural advertising purposes.
</item 56>

<item 57>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
The Commission designated 19 VLOPs back in April — including Meta owned Facebook and Instagram.

The designation puts obligations on VLOPs to respond diligently to reports of illegal content, as well as clearly communicate their T&Cs to users and properly
enforce their terms. But it also ranges more widely — requiring these larger platforms to take steps to identify and mitigate systemic risks such as
disinformation.

The regulation also contains a “crisis response” mechanism which the Commission may adopt on VLOPs in situations where use of their platforms could contribute
to serious threats such as war. Although this aspect of the regulation won’t start operating until early next year (Feburary 17) as it requires an advisory
body, called the European Board for Digital Services, to be operational so it can recommend the Commission to adopt a crisis response mechanism (and the set up
of the Board is waiting on Member States to establish national Digital Services Coordinators, aka the bodies that will oversee DSA compliance for non-VLOPs).
</item 57>

<item 58>
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
</item 58>

<item 59>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Following a content moderation warning from European Union regulators earlier this week, Meta has published an overview of how its responding to risks on its
social media platforms stemming from the Israel-Hamas war.

Its blog post covers what it frames as “ongoing efforts”, with some existing policies and tools for users rehashed. But the company confirms it’s made some
changes in light of fast moving events in Israel and Gaza.

These include what it says is a temporary expansion of its Violence and Incitement policy in order to prioritize the safety of Israelis kidnapped by Hamas.

Under this change Meta says it will be removing content that “clearly identifies hostages when we’re made aware of it, even if it’s being done to condemn or
raise awareness of their situation”. “We are allowing content with blurred images of the victims but, in line with standards established by the Geneva
Convention, we will prioritize the safety and privacy of kidnapping victims if we are unsure or unable to make a clear assessment,” it added.
</item 59>

<item 60>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
They choose to do things that maximize the frequency of use of that platform by children, and the duration of that use. They could end all this today if they
wanted, they could easily keep those under 13 from accessing their platform. But they’re not.”

You can read the mostly unredacted complaint here.

(This story has been updated with a comment from Meta.)
</item 60>

<item 61>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
So Meta’s updated offer to EU users is either hand over your privacy or hand over your hard earned cash.

“This is an unfair choice for users, which runs afoul of EU consumer law on several counts and must be stopped,” said the European Consumer Organisation (BEUC)
in a press release announcing the complaint will be filed with the network of consumer protection authorities (CPC) today.

BEUC has been joined in the complaint by 18 of its member organizations — a variety of consumer advocacy groups which are located in the following EU member
states: Bulgaria, Czech Republic, Denmark, France, Greece, Italy, Latvia, Lithuania, Netherlands, Norway, Poland, Slovakia, Slovenia, Spain and Sweden.

The groups are objecting both to how Meta has gone about implementing the “pay-or-consent model” — using what they assess as “unfair, deceptive and aggressive
practices” — and to the model itself, which they dub “illegal”.
</item 61>

<item 62>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.
</item 62>

<item 63>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
Today’s announcement by the EU, following several months of in-depth examination looking at how the transaction might impact competition, lays out formal
concerns for Amazon to respond to. The EU’s objections focus on foreclosure risks.

The EU says its concerns focus on whether the deal will give Amazon the ability and incentive to foreclose iRobot’s rivals by deploying strategies aimed at
preventing competitors from selling RVCs on Amazon’s marketplace and/or making it harder for them to do so — such as by delisting rival products; reducing their
visibility in both non-paid (i.e. organic) and paid results (i.e. ads) displayed on its marketplace; limiting their access to certain widgets it operates (such
as the “other products you may like” feature) or “certain commercially-attractive product labels” (e.g. ‘Amazon’s choice’ or ‘Works With Alexa’); and/or by
directly or indirectly raising the costs of iRobot’s rivals to advertise and sell their RVCs on its marketplace.
</item 63>

<item 64>
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
</item 64>

<item 65>
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
</item 65>

<item 66>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Per Mekić, the Commission’s ad campaign, which apparently racked up millions of views on X, insinuated that opponents of the proposed legislation did not want
to protect children — messaging he dubbed “a form of emotional blackmail”.

The ads included what he suggested is a misleading claim that the majority of Europeans support the proposal — but which is based on a survey that highlighted
“only the benefits but not the drawbacks of the proposed legislation”. Other surveys, by research firms YouGov and Novus, that highlighted the drawbacks showed
“virtually no support” for the plan among the European population, his post also pointed out.
</item 66>

<item 67>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 67>

<item 68>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
The Commission is the sole enforcer of the DMA.

Meta is already in the Commission’s sights over its approach to the DSA — with the EU’s executive, in recent days, requesting more information from the tech
giant vis-à-vis its approach to content threats flowing from the Israel-Hamas war and about election security issues. But it remains to be seen whether the EU
will apply the same close scrutiny to Meta’s ad tracking offer.

In its blog post Meta claims the choice to offer people to pay it for their privacy or else agree to being tracked “balances the requirements of European
regulators while giving users choice and allowing Meta to continue serving all people in the EU, EEA and Switzerland”. But, well, it would say that.

The subscription is also only being offered to people aged 18 and up — which raises questions over how it will comply with requirements in the DSA and DMA not
to processing children’s data for ad targeting.
</item 68>

<item 69>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Although the DSA’s provisions against dark patterns are only intended to be applied where consumer protection and privacy laws, which also take aim against
unfair choices, don’t.
</item 69>

<item 70>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
During this period the DSA had not entered into force yet for VLOPs and VLOSEs, including X/Twitter. The Commission cannot comment further on the substance of
the risk assessments. We are analysing the available information on a range of issues and we are in close contact with X but we cannot comment on specifics and
timelines. As stated by Commissioner [Thierry] Breton, minor protection and integrity of elections are priorities in the enforcement of the DSA.

Penalties for breaches of the DSA can scale up to 6% of global annual turnover and the Commission has previously wanted that not following the Code of Practice
on Disinformation would be factored into its assessments. So the price for X ignoring the EU’s warnings on disinformation could be steep — albeit its ad revenue
has cratered since Musk’s takeover, as a result of his failure to tackle toxicity and harmful disinformation encouraging an advertiser exodus — and shrinking
earnings would effectively downsize any future penalties X could face under the DSA.
</item 70>

<item 71>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
And also today, as we reported earlier, Jourova drew attention to early analysis conducted by some of the remaining signatories which she said had found X
performed the worst for disinformation ratios.

This suggests that X, which back in April was designated by the EU as a VLOP under the DSA, continues to put itself squarely in the Commission’s crosshairs —
including over its priority issue of tackling Kremlin propaganda.

As well as devising the anti-disinformation Code, the bloc’s executive is now responsible for oversight of VLOPs’ compliance with the DSA — with powers under
the new law to fine violators up to 6% of global annual turnover.
</item 71>

<item 72>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
“In the three days following October 7, we removed or marked as disturbing more than 795,000 pieces of content for violating these policies in Hebrew and
Arabic,” it wrote. “As compared to the two months prior, in the three days following October 7, we have removed seven times as many pieces of content on a daily
basis for violating our Dangerous Organizations and Individuals policy in Hebrew and Arabic alone.”

In light of the dialled up attention on and concern about the situation, Meta says it’s possible non-violating content may be removed “in error”.

“To mitigate this, for some violations we are temporarily removing content without strikes, meaning these content removals won’t cause accounts to be disabled,”
it notes. “We also continue to provide tools for users to appeal our decisions if they think we made a mistake.”

Compliance with the bloc’s Digital Services Act (DSA) kicked in for Meta in August as the owner of a so-called very large online platform (VLOP).
</item 72>

<item 73>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
But the result left it unclear how cookie paywalls might be operated in a way that’s GDPR compliant and the privacy rights group vowed to fight the decision in
court. “The final decision on ‘pay or okay’ may be made by the European Court of Justice (CJEU) in the long run,” noyb predicted at the time.

Meta is likely banking on another multi year round of GDPR complaints, legal challenges and — finally — a referral to the CJEU, followed by another long wait
before a ruling gets handed down, buying it several more years to run with its new legal basis fix and keep feeding its profits by doing what it likes with
Europeans’ data.

But the consumer protection challenge could complicate its usual playbook.

The CPC has brought more coordinated action on consumer protection concerns in the EU in recent years, bringing multiple consumer groups together to tackle
common concerns — helped by one or more national consumer protection authority which gets appointed to drive the effort.
</item 73>

<item 74>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 74>

<item 75>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.
</item 75>

<item 76>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
The promotion of our proposal is standard normal practice

3. This proposal is about protecting children from sexual abuse https://t.co/zSSAu3684P — Ylva Johansson (@YlvaJohansson) October 13, 2023

“When I made a tweet on the 100% legal [point] that was based on the information I had. But I have to be very open; then I got other information that there
could be question marks on the compliance with the DSA — and I take this very seriously,” she told the committee. “If that is the case then of course there has
to be consequences on that. So that’s why it’s important that we have to look into [it]. Of course we always have to comply with the regulation. There’s no
question about that.”
</item 76>

<item 77>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
They have also raised data protection concerns which are already the focus of the complaint sent to the Austrian data protection authority earlier this week by
the privacy rights not-for-profit, noyb.

Commenting in a statement, Ursula Pachl, deputy director general of BEUC, said:

The choice the tech giant is currently providing to consumers is unfair and illegal — the millions of European users of Facebook and Instagram deserve far
better than this. Meta is breaching EU consumer law by using unfair, deceptive and aggressive practices, including partially blocking consumers from using the
services to force them to take a decision quickly, and providing misleading and incomplete information in the process. Consumer protection authorities in the EU
must now spring into action and force the tech giant to stop this practice.

Summarizing the issues identified with Meta’s model under consumer protection law BEUC writes:

Meta is partially blocking the use of Facebook and Instagram until users have selected one option or the other, which constitutes an aggressive practice under
European consumer law.
</item 77>

<item 78>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Asked whether the issue for consumer protection authorities is the ‘pay or consent’ choice Meta is offering or how it’s gone about implementing it, BEUC’s
spokesman said the questions are hard to separate as they’re “closely interlinked”.

“Under consumer law, you need an informed and fair choice to purchase such a subscription. The first question is also dependent on compliance with data
protection law. If the practice infringes the GDPR, the fact that it infringes a law which aims to protect fundamental rights should in our opinion be
considered unfair and illegal under consumer law too,” he told us, adding: “In any case, the choice is designed in a way that is unfair, aggressive and
misleading.”

The European Commission itself has an additional oversight role on Meta directly as the company is also subject to the EU’s shiny new Digital Markets Act (DMA)
and Digital Services Act (DSA).
</item 78>

<item 79>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
The Commission risks straying close to making attacks on journalists by using such tactics, Fotiadis warned — adding: “You cannot just dismiss everything by
calling fake news” — before also noting that Johansson’s office had declined multiple interview requests ahead of publication of the article.

Responding to a question from the committee about the reporting he said documents obtained by the journalists included email threads between Commission
officials in Johansson’s department, DG-Home, and a “key stakeholder” advocating for the use of technology for CSAM-scanning — which indicated what he described
as “privileged access” that “speaks directly to cooperation” and goes “way beyond” mere consultation or exchange of views on the proposal.

“It’s an official chain discussing invitation, how the stakeholder would be able to allocate experts that would speak in workshops — first attended by
representatives of the Member States, and then afterwards actually by ministers in the Council in a meeting chaired by commissioner Johansson.
</item 79>

<item 80>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.
</item 80>

<item 81>
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
</item 81>

<item 82>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
These included findings made by the Court of Justice of the European Union when delivering judgment on July 4, 2023 in a case in which the Court examined the
legal bases on which Meta’s processing of users’ data for behavioural advertising purposes is grounded.

“Acting in consultation with its fellow European supervisory authorities, the DPC has been engaged in a detailed regulatory assessment of the consent-based
model since it was first proposed by Meta in July. That exercise is being led by the DPC, reflecting its position as Lead Supervisory Authority for Facebook and
Instagram in Europe. The exercise has not yet concluded, and no findings have been made to date. It is due to be completed shortly, at which point the DPC will
notify Meta if it considers that the manner in which its new user offerings are to be implemented is compatible with Meta’s obligations under GDPR.”

So Meta’s move to offer users a subscription or tracking hasn’t been signed off by data protection authorities as yet — and it’s clear that further regulatory
interventions could follow.
</item 82>

<item 83>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Through persistence and by creating a sense of urgency, Meta pushes consumers into making a choice they might not want to take.

In addition, many consumers likely think that, by opting for the paid subscription as it is presented, they get a privacy-friendly option involving less
tracking and profiling. In fact, users are likely to continue to have their personal data collected and used, but for purposes other than ads.

Meta provides misleading and incomplete information to consumers which does not allow them to make an informed choice. Meta is misleading them by presenting the
choice as between a paying and a ‘free’ option, while the latter option is not ‘free’ because consumers pay Meta through the provision of their data, as past
court rulings have already declared.
</item 83>

<item 84>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
In a very swift test of the European Union’s newly updated content moderation rulebook, the bloc has fired a public warning at Elon Musk–owned X (formerly
Twitter) for failing to tackle illegal content circulating on the platform in the wake of Saturday’s deadly attacks on Israel by Hamas terrorists based in the
Gaza Strip.

The European Commission has also raised concerns about the spread of disinformation on X related to the terrorist attacks and their aftermath.

Unlike terrorism content, disinformation is not illegal in the EU per se. However, the EU’s Digital Services Act (DSA) puts an obligation on X — as a so-called
very large online platform — to mitigate risks attached to harmful falsehoods as well as to act diligently on reports of illegal content.

Following the terrorist attacks by Hamas against 🇮🇱, we have indications of X/Twitter being used to disseminate illegal content & disinformation in the EU.
</item 84>

<item 85>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 85>

<item 86>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Confirmed breaches of the online governance regime can face a range of major sanctions, including fines of up to 6% of global annual turnover.

The EU can also apply interim measures where it believes there’s a risk of serious harm for users and may even seek to have access to infringing services
temporarily blocked. Although its announcement today is careful to caveat the development by affirming no conclusions have yet been reached.

There is also no confirmed timeline for the probe to conclude. But the fact of it being opened means other tools are put within reach of regulators and could be
applied if the EU sees the need to act quickly — such as the aforementioned interim measures. So the Commission’s official scrutiny of X could have real world
implications for how the platform operates sooner rather than later.
</item 86>

<item 87>
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
</item 87>

<item 88>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
A newly unredacted version of the multi-state lawsuit against Meta alleges a troubling pattern of deception and minimization in how the company handles kids
under 13 on its platforms. Internal documents appear to show that the company’s approach to this ostensibly forbidden demographic is far more laissez-faire than
it has publicly claimed.

The lawsuit, filed last month, alleges a wide spread of damaging practices at the company relating to the health and well-being of younger people using it. From
body image to bullying, privacy invasion to engagement maximization, all the purported evils of social media are laid at Meta’s door — perhaps rightly, but it
also gives the appearance of a lack of focus.

In one respect at least, however, the documentation obtained by the attorneys general of 42 states is quite specific, “and it is damning,” as AG Rob Bonta of
California put it. That is in paragraphs 642 through 835, which mostly document violations of the Children’s Online Privacy Protection Act, or COPPA.
</item 88>

<item 89>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“We can tailor our ads to hit exactly the consumers we want,” Moller said. “It’s a perfect marriage of a transaction, and that’s one of the beauties of what
their marketplace offers.” Needham said he feels he would have been more supportive of the case if it were filed a few years ago, pre-pandemic. At that time, he
said, "I would have felt, yes Amazon is a monopoly… But actually after Covid, into 2023, ecommerce has had a lot of big changes.” He added, "The competition is
just not what it was in 2019." Competitors like Shopify and Walmart are increasingly viable alternatives for many categories of sellers, Needham said, not to
mention rapidly growing Chinese e=commerce companies like Temu. As a result, Needham said he’s seen some significant changes from Amazon. Among those is a
greater ability for Amazon sellers to communicate with buyers, offering select customers certain promotions.
</item 89>

<item 90>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
During an exchange of views with the European Parliament’s civil rights, justice and home affairs (LIBE) committee this afternoon, Johansson admitted the EU’s
executive is investigating the matter.

Initially she had sought to dismiss criticism over the legality of the microtargeting — claiming in a tweet earlier this month (embedded below) that the
campaign was “100%” legal.

Asked by the LIBE committee about the discrepancy between her tweet and the existence of an investigation Johansson said she had been given “new information”
related to DSA compliance that merited looking into.

As my services have been directly accused of illegal acts👇 I think it is import I step in:

1. @EUHomeAffairs have followed the guidelines & the law 100%

2. The promotion of our proposal is standard normal practice

3.
</item 90>

<item 91>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
“Amazon may have the ability to foreclose iRobot’s rivals because Amazon’s online marketplace is a particularly important channel to sell RVCs in France,
Germany, Italy, and Spain,” the Commission writes in a press release. “RVC customers in these countries particularly rely on Amazon both in terms of product
discovery as well as for their final purchasing decision.”

The Commission is also worried Amazon may have the incentive to foreclose iRobot’s rivals because it may be economically profitable for it to do so. “The merged
entity would likely gain more from additional sales of iRobot RVCs, than it would lose from fewer sales of iRobot’s rivals and other related products on Amazon.
Such gains include benefits from additional data gathered from iRobot’s users,” it suggests.

Were Amazon to deploy such foreclosure strategies the Commission’s view at this stage of the probe is it could restrict competition in the market for RVCs,
leading to “higher prices, lower quality, and less innovation for consumers”.
</item 91>

<item 92>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Commitment 15 in TikTok’s report signs the platform up to “tak[ing] into consideration transparency obligations and the list of manipulative practices
prohibited under the proposal for Artificial Intelligence Act” — and here it lists being a launch partner of the Partnership on AI’s (PAI) “Responsible
Practices for Synthetic Media” (and contributing to the development of “relevant practices”); and joining “new relevant groups”, such as the Generative AI
working group which started work this month as implemented measures towards this pledge.

In the next six months it says it wants to further strengthen its enforcement of its synthetic media policy — and explore “new products and initiatives to help
enhance our detection and enforcement capabilities” in this area, including in the area of user education.

Meta

Facebook and Instagram parent Meta’s report also includes a recognition that “widespread availability and adoption of generative AI tools may have implications
for how we identify, and address disinformation on our platforms”.
</item 92>

<item 93>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Penalties for failing to comply with the pan-EU regulation can reach as high as 6% of global annual turnover — which, in Meta’s case, could run to multiple
billions.

The social media giant is not alone in being warned by the bloc over content concerns attached to the Israel-Hamas war: Elon Musk’s X has been singled out for
even greater attention here — with the bloc issuing both an “urgent” warning earlier this week and following that with a formal request for information about
its compliance approach.

TikTok has also received a warning from the EU about DSA content risks related to the conflict.

This report was updated with details about the operational timeline for the DSA crisis response mechanism coming into operation
</item 93>

<item 94>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Going into more detail of the microtargeting used by the Commission, Mekić wrote: “X’s Transparency Report shows that the European Commission also used
‘microtargeting’ to ensure that the ads did not appear to people who care about privacy (people interested in Julian Assange) and eurosceptics (people
interested in ‘nexit’, ‘brexit’ and ‘spanexit’ or in Victor Orbán, Nigel Farage, or the German political party AfD). For unclear reasons, people interested in
Christianity were also excluded.

“After excluding critical political and religious groups, X’s algorithm was set to find people in the remaining population who were indeed interested in the ad
message, resulting in an uncritical echo chamber. This microtargeting on political and religious beliefs violates X’s advertising policy, the Digital Services
Act – which the Commission itself has to oversee — and the General Data Protection Regulation [GDPR].”
</item 94>

<item 95>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.
</item 95>

<item 96>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
“The OpenAI / Microsoft drama underlines one of the big near-term risks with AI — that this next wave of technology is controlled by the same tiny set of
players who have shaped that last era of the internet,” Mark Surman, president and executive director at the Mozilla Foundation, told TechCrunch. “We might have
a chance of avoiding this if GPT-X were responsibly open sourced, giving researchers and startups a shot at making this technology safer, more useful and more
trustworthy for people everywhere.”

Open and shut

In an open letter published by Mozilla a few weeks back, Meta’s chief AI scientist Yann LeCun joined some 70 other signatories in calling for more openness in
AI development, though that letter has since garnered more than 1,700 signatures. The backdrop stems from Big Tech companies such as OpenAI and Google’s
DeepMind calling for more regulation, warning of catastrophic consequences if the AI levers were to meet the wrong hands — in other words, they argued that
proprietary AI is safer than open source.
</item 96>

<item 97>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
But it has yet to communicate a conclusion. In the meanwhile, Meta maintains that the model it has devised for obtaining users’ consent to its ads processing
complies with the General Data Protection Regulation (GDPR). (Although the adtech giant also said that when it was claiming performance of a contract and then
legitimate interests for the processing — both of which were subsequently found to be incompatible with the GDPR.)

The ‘pay or okay’ model Meta is seeking to impose on EU users wasn’t actually its invention; it was ‘pioneered’ in Austria, by the daily newspaper Der Standard
— after which copycat cookie paywalls quickly sprung up on a raft of news publishers in Germany and elsewhere in the EU.

noyb has been challenging this ‘pay or okay’ approach to GDPR consent since 2021— filing complaints with a number of data protection authorities, arguing the
model forces newspaper readers to “buy back their own data at exorbitant prices”.
</item 97>

<item 98>
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
</item 98>

<item 99>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 99>

<item 100>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 100>
=== END ITEM id=c088cee25cb741c8 ===

=== ITEM id=1503bd5a7de18168 ===
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

=== END ITEM id=1503bd5a7de18168 ===

=== ITEM id=676df71abd9d1f34 ===
Gold sentences and facts:
Sentence 0: They’d all have significant value if president of baseball operations Jerry Dipoto decided to move one for a bat.
  Fact 0.0: Jerry Dipoto is president of baseball operations.
  Fact 0.1: Those referred to as 'they' would all have significant value if Jerry Dipoto decided to move one for a bat.
Sentence 1: Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.
  Fact 1.0: Aaron Judge came away with a new contract.
  Fact 1.1: Justin Verlander came away with a new contract.
  Fact 1.2: Trea Turner came away with a new contract.
  Fact 1.3: Xander Bogaerts came away with a new contract.

Retrieved items:

=== END ITEM id=676df71abd9d1f34 ===
