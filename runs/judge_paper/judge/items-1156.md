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

This file has 2 items.

=== ITEM id=833791475b7b01aa ===
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
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
It replied with an automated email reading: “Busy now, please check back later.”

Update: X’s global government affairs account has posted a thread responding to the EU’s criticism in which it claims the data shows other social networking
services saw greater changes in subscriber growth and total growth for pro-Kremlin accounts, and for engagement and volume; it also writes that: “More than 700
unique Community Notes have appeared on posts related to the Ukraine conflict. Through Community Notes’ new ‘Notes on Media’ feature, which automatically
matches notes on an image or video to other posts that contain matching media, these notes appeared on over 1,200 additional posts. These notes are in numerous
languages and are written specifically for local audiences.”

Back in April, X/Twitter was named as one of 19 larger platforms that face the strictest level of regulation under the EU’s DSA.
</item 1>

<item 2>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta forbids support of terrorist organizations, like most social media companies based in the U.S., but struggles to moderate content around it, from user
discourse to journalistic updates. This policy, along with the company’s partnership with Israel to monitor posts that incite violence, complicates things for
Palestinians living under Hamas’ governance. As EFF points out, something as simple as Hamas’ flag in the background of an image can result in a strike.

Jillian York, the director for international freedom of expression for EFF, blames automation and decisions made by “minimally trained humans” for the
inconsistency. Meta’s zero tolerance policy and imprecise enforcement often suppress content from or about conflict zones, she said. The site’s moderation
issues have negatively affected multiple non-English speaking regions, including Libya, Syria and Ukraine.

“These rules can prevent people from sharing documentation of human rights violations, documentation of war crimes, even just news about what’s happening on the
ground,” York continued.
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
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
But some might argue that it’s unhealthy that so much focus lies on just a handful of people — and it’s telling that their departure has created such widespread
havoc.
</item 4>

<item 5>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
In this way, Meta goes through great lengths to avoid meaningfully complying with COPPA, looking for loopholes to excuse its knowledge of users under the age of
13 and maintain their presence on the Platform.

The other complaints in the lengthy lawsuit have softer edges, such as the argument that use of the platforms contributes to poor body image and that Meta has
failed to take appropriate measures. That’s arguably not as actionable. But the COPPA stuff is far more cut and dry.

“We have evidence that parents are sending notes to them about their kids being on their platform, and they’re not getting any action. I mean, what more should
you need? It shouldn’t even have to get to that point,” Bonta said.

“These social media platforms can do anything they want,” he continued. “They can be operated by a different algorithm, they can have plastic surgery filters or
not have them, they can give you alerts in the middle of the night or during school, or not.
</item 5>

<item 6>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In this case, failure to reply by the deadline could lead to the imposition of period penalties,” the EU added.

Earlier this year, Musk pulled out of the EU’s Code of Practice on online disinformation. In a response to X leaving the voluntary Code, Breton warned:
“Obligations remain. You can run but you can’t hide.”

In recent days, TikTok and Meta have also been warned by the EU over disinformation related to the Israel-Hamas war. But back in September, a study commissioned
via the EU’s Code of Practice suggested X is the worst of the major platforms when it comes to spreading disinformation.

This report was substantially revised to clarify a number of aspects, including — most saliently — that the EU has sent a formal request for information to X at
this point. This could prefigure a formal announcement of an investigation but is not, yet, that formal technical step.
</item 6>

<item 7>
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
</item 7>

<item 8>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.
</item 8>

<item 9>
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
</item 9>

<item 10>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Going into more detail of the microtargeting used by the Commission, Mekić wrote: “X’s Transparency Report shows that the European Commission also used
‘microtargeting’ to ensure that the ads did not appear to people who care about privacy (people interested in Julian Assange) and eurosceptics (people
interested in ‘nexit’, ‘brexit’ and ‘spanexit’ or in Victor Orbán, Nigel Farage, or the German political party AfD). For unclear reasons, people interested in
Christianity were also excluded.

“After excluding critical political and religious groups, X’s algorithm was set to find people in the remaining population who were indeed interested in the ad
message, resulting in an uncritical echo chamber. This microtargeting on political and religious beliefs violates X’s advertising policy, the Digital Services
Act – which the Commission itself has to oversee — and the General Data Protection Regulation [GDPR].”
</item 10>

<item 11>
Discord’s latest teen safety blitz starts with content filters and automated warnings (Engadget, 2023-10-19T15:00:31+00:00)
Discord is introducing new safeguards to protect young users. On Thursday, the company announced Teen Safety Assist. The rollout will happen in the coming weeks
and consists of two features: Automated alerts and content filters.

Moving forward, when a user a teen hasn’t chatted with before sends a direct message, Discord will display a safety alert if the platform’s automated systems
suspect the teen’s safety could be at risk. The prompt will nudge the young person to reconsider if they want to respond to the message, and point them to the
app’s block feature and related safety tips.

By default, Discord will now also automatically blur potentially sensitive images from a teen’s friends. Young people can disable the feature through an option
in Discord’s settings menu. Adults, meanwhile, can enable the filters for themselves, if they want the additional protections.

The introduction of Teen Safety Assist comes amid increased scrutiny of Discord’s efforts to limit the spread of child sexual abuse material (CSAM) and protect
young users from predatory adults.
</item 11>

<item 12>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
Even for a user with just one account (on either Facebook or Instagram) the cost for protecting their privacy from Meta’s tracking and profiling would be almost
€120 per year (for web use) or just over €155 (on mobile).

As we reported earlier this month, Meta is relying on a line in a ruling handed down by the bloc’s top court, the CJEU, earlier this year — where the judges
allowed the possibility — caveated with “if necessary” — of an (another caveat) “appropriate fee” being charged for an equivalent alternative service (i.e. that
lacks tracking and profiling). So the legal fight against Meta’s continued tracking and profiling of users will hinge on what’s necessary and appropriate in
this context.

noyb, the European privacy rights group which has driven much of the strategic litigation against Meta’s tracking and profiling, has already — since 2021 — been
challenging similar ‘pay or okay’ practices by news publishers by filing a series of complaints with data protection authorities.
</item 12>

<item 13>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
His argument essentially was that the improvements and safety features Meta was implementing were not enough and not effective.

However, a policy position like the one Meta announced today is not something that would have been rushed out in a matter of days as a PR response to Bejar’s
claims, but something Meta has been stewing on for some time. After all, the app stores already have age guidelines, so it would seem to follow that they should
also enforce them, if enforcement is to be regulated — or that seems to be Meta’s thinking.

If anything, the call to bring the app stores into the fight is a response to the lack of action in Congress to pass online child safety regulations, like the
hotly debated KOSA (the Kids Online Safety Act) leaving U.S. states to individually create laws due to a void of federal guidelines.

Utah, for instance, passed a measure that will require social networks to give Utah parents access to their children’s posts, messages and comments, and would
block social media access during certain hours.
</item 13>

<item 14>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
And these actions reportedly don’t cross between platforms, meaning an Instagram account being disabled doesn’t flag associated or linked Facebook or other
accounts.

Zuckerberg testified to Congress in March of 2021 that “if we detect someone might be under the age of 13, even if they lied, we kick them off.” (And “they lie
about it a TON,” one research director said in another quote.) But documents from the next month cited by the lawsuit indicate that “Age verification (for under
13) has a big backlog and demand is outpacing supply” due to a “lack of [staffing] capacity.” How big a backlog? At times, the lawsuit alleges, on the order of
millions of accounts.

A potential smoking gun is found in a series of anecdotes from Meta researchers delicately avoiding the possibility of inadvertently confirming an under-13
cohort in their work.

One wrote in 2018: “We just want to make sure to be sensitive about a couple of Instagram-specific items.
</item 14>

<item 15>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
And it isn’t because leadership wasn’t aware.

That same year, 2018, CEO Mark Zuckerberg received a report that there were approximately 4 million people under 13 on Instagram in 2015, which amounted to
about a third of all 10-12-year-olds in the U.S., they estimated. Those numbers are obviously dated, but even so they are surprising. Meta has never, to our
knowledge, admitted to having such enormous numbers and proportions of under-13 users on its platforms.

Not externally, at least. Internally, the numbers appear to be well documented. For instance, as the lawsuit alleges:

Meta possesses data from 2020 indicating that, out of 3,989 children surveyed, 31% of child respondents aged 6-9 and 44% of child respondents aged 10 to
12-years-old had used Facebook.
</item 15>

<item 16>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Most recently, Instagram users accused Meta of “shadowbanning” their Stories about Palestine.

It’s the latest in a lengthy history of incidents on Meta platforms that reflect an inherent bias against Palestinian users in its processes, as documented by
years of complaints from both inside and outside the company. The company may not intentionally suppress specific communities, but its moderation practices
often disproportionately affect Palestinian users.

For instance, Meta struggles to navigate the cultural and linguistic nuances of Arabic, a language with over 25 dialects, and has been criticized for neglecting
to adequately diversify its language resources. The company’s black-and-white policies often preclude it from effectively moderating any nuanced topic, like
content that discusses violence without condoning it. Advocacy groups have also raised concerns that Meta’s partnerships with government agencies, such as the
Israeli Cyber Unit, politically influence the platform’s policy decisions.
</item 16>

<item 17>
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
</item 17>

<item 18>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.
</item 18>

<item 19>
X plans to use payment, phone and ID verification to stop bots, in addition to the new $1/yr fee (TechCrunch, 2023-10-18T17:33:48+00:00)
A report this week from market intelligence firm Similarweb, for example, found that X’s web traffic and monthly active users on mobile had declined
post-acquisition.

Users today have far more options for microblogging and public discourse, thanks to products from Twitter-like startups and tech companies such as Spill,
Bluesky, Pebble, Countersocial, Spoutible, Hive and the open source platform Mastodon, as well as efforts from tech giants, like Meta’s new app Threads. Few of
the other social apps currently charge a fee, unless it’s for premium features, like some of the third-party Mastodon apps do.

In addition to presenting friction at sign-up, other critics have argued a fee could exclude genuine users and increase the digital divide.
</item 19>

<item 20>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Following a content moderation warning from European Union regulators earlier this week, Meta has published an overview of how its responding to risks on its
social media platforms stemming from the Israel-Hamas war.

Its blog post covers what it frames as “ongoing efforts”, with some existing policies and tools for users rehashed. But the company confirms it’s made some
changes in light of fast moving events in Israel and Gaza.

These include what it says is a temporary expansion of its Violence and Incitement policy in order to prioritize the safety of Israelis kidnapped by Hamas.

Under this change Meta says it will be removing content that “clearly identifies hostages when we’re made aware of it, even if it’s being done to condemn or
raise awareness of their situation”. “We are allowing content with blurred images of the victims but, in line with standards established by the Geneva
Convention, we will prioritize the safety and privacy of kidnapping victims if we are unsure or unable to make a clear assessment,” it added.
</item 20>

<item 21>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What does this have to do with the First Amendment?

The case revolves around First Amendment rights — but, counterintuitively, it’s the rights of social media companies that are in question, not the rights of
their users.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment?”

“These cases are about the First Amendment and how the First Amendment applies to social media companies. And then more specifically, what the First Amendment
has to say about content moderation, which is obviously a subset, although a really big and important subset of what social media companies do,” Barrett said.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment — or what you also
might call content moderation — in sorting out what expression does and does not appear on the platforms that they own? So it’s just not clear right now what
precisely the First Amendment says about that question.”

Are the Texas and Florida laws identical?
</item 21>

<item 22>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“As if there’s better moderation for those languages, and more careless moderation for Arabic content.”

Stone declined to comment on allegations of Meta disproportionately flagging Arabic news content, instead directing TechCrunch to the updated newsroom post —
which did not directly address the allegations.

And as the Intercept reported, Instagram and Facebook are flagging images of the al-Ahli Hospital, claiming that the content violates Meta’s Community
Guidelines on nudity or sexual activity.

The Community Guidelines are enforced inconsistently, particularly when it comes to content related to Palestine. Al Saadi recently tried to report a comment
that said she should be “raped” and “burned alive” — left in response to her comment on a CNN post about the conflict — but in screenshots reviewed by
TechCrunch, Instagram said that it didn’t violate the platform’s Community Guidelines against violence or dangerous organizations.

“The restrictions on content, especially the content that relates to Palestine, is heavily politicized,” Abukhater said. “It feeds into the bias against
Palestinian narrative genuinely.
</item 22>

<item 23>
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
</item 23>

<item 24>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 24>

<item 25>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
Later that year, external researchers provided Meta with information that “of children ages 9-12, 45% used Facebook and 40% used Instagram daily.”

During an internal 2021 study on youth in social media described in the suit, they first asked parents if their kids are on Meta platforms and removed them from
the study if so. But one researcher asked, “What happens to kids who slip through the screener and then say they are on IG during the interviews?” Instagram
Head of Public Policy Karina Newton responded, “we’re not collecting user names right?” In other words, what happens is nothing.

As the lawsuit puts it:

Even when Meta learns of specific children on Instagram through interviews with the children, Meta takes the position that it still lacks actual knowledge of
that it is collecting personal information from an under-13 user because it does not collect user names while conducting these interviews.
</item 25>

<item 26>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“The first is with their legal occupier. The second war is with the entire Western media landscape, and when I say the entire landscape, I mean social media.”

Meta’s alleged shadowbanning

Instagram users accuse Meta of suppressing more than just Stories related to Palestine.

Creators say engagement on their posts tanked specifically after they publicly condemned Israel’s response to the Hamas attack as excessively violent. Some,
like Jabr, say they were restricted from posting or going live, while others say Instagram flagged their content as “sensitive,” limiting its reach. Users also
allege their posts were flagged as “inappropriate” and removed, even if the content adhered to Instagram’s Community Guidelines.

Meta’s representative didn’t address the other accusations of censorship beyond just Story visibility and did not respond to TechCrunch’s follow-up questions.
When asked for comment, Stone directed TechCrunch to Meta’s updated newsroom post, which said Meta identified and fixed a “bug” that prevented users from going
live.
</item 26>

<item 27>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
It’s difficult to extrapolate from the 2015 and 2020 numbers to today’s (which, as we have seen from the evidence presented here, will almost certainly not be
the whole story), but Bonta noted that the large figures are presented for impact, not as legal justification.

“The basic premise remains that their social media platforms are used by millions of children under 13. Whether it’s 30 percent, or 20 or 10 percent… any child,
it’s illegal,” he said. “If they were doing it at any time, it violated the law at that time. And we are not confident that they have changed their ways.”

An internal presentation called “2017 Teens Strategic Focus” appears to specifically target kids under 13, noting that children use tablets as early as 3 or 4,
and “Social identity is an Unmet need Ages 5-11.”
</item 27>

<item 28>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
“Meta notified the DPC on July 27 of its intention to implement an alternative, consent-based model in which users would be given a choice between ads-funded
versions of its platforms, and subscription versions in which, in return for payment of monthly fee, it is said that users will not receive targeted
advertising,” the Irish regulator wrote.

“While February 2024 was initially identified by Meta as the earliest date by which its consent model would be operational, it agreed to bring forward that date
to November 2023 at the direction of the DPC, the DPC having been concerned to ensure that changes would be implemented to the platforms as soon as practicable
in light of earlier findings to the effect that Meta had failed to demonstrate its entitlement to rely on the legal bases on which it was then relying when
processing users’ data for behavioural advertising purposes.
</item 28>

<item 29>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Critics on both the right and the left described the launch as a disaster.

Conservative organizations have proven especially out of their depth when facing the wrath of extremely online social media users, who mobilize trolling for
social justice. In 2020, TikTok users reserved hundreds of tickets for a Trump rally and never showed up, leaving the president to face swaths of empty seats.
When a Texas anti-abortion group opened an anonymous tip form to enforce the state’s ban on abortion after six weeks, Gen Z activists flooded the site with
Shrek porn. This year, TikTok and Twitter users shut down the Missouri attorney general’s tip form for reporting clinics that provide gender-affirming care.
Within days of launching, the site was spammed with fanfiction, satirical anecdotes about kids getting “transed” and fanfiction. The attorney general’s press
secretary blamed the site’s downfall on activists “hacking the system.”
</item 29>

<item 30>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
During the last violent outbreak between Hamas and Israel in 2021, a report commissioned by Meta and conducted by a third party concluded that the company’s
actions had an “adverse human rights impact” on Palestinian users’ right to freedom of expression and political participation.

Meta communications director Andy Stone declined to comment on the record, and pointed TechCrunch to Meta’s newsroom post about the conflict, which was updated
on Oct. 18 with a statement addressing the shadowbanning claims.

“Our policies are designed to keep people safe on our apps while giving everyone a voice,” the post stated. “We apply these policies equally around the world
and there is no truth to the suggestion that we are deliberately suppressing voice … We can make errors and that is why there is an appeals process for people
to tell us when they think we have made the wrong decision, so we can look into it.”

The belief that Meta shadowbans, or limits the visibility of, content about Palestine is not new.
</item 30>

<item 31>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“It happened to all my friends, too,” she continued. “Then we noticed that posting a random pic would get higher views. So by posting a random pic, then a
pro-Palestine post, would increase the views.”

Another Instagram user based in the United Kingdom, who also asked to stay anonymous out of fear of harassment, said that his view count returned to normal when
he posted a cat photo.

“My stories went from 100s of views to zero or a handful,” he said. “I’ve had to post intermittent non-Gaza content in order to ‘release’ my stories to be
viewed again.”

It isn’t just Stories. The Arab Center for Social Media Advancement (7amleh), which documents cases of Palestinian digital rights violations and works directly
with social media companies to appeal violations, told TechCrunch it has received reports of Instagram inconsistently filtering comments containing the
Palestinian flag emoji. Users report that Instagram has flagged comments containing the emoji as “potentially offensive,” hiding the comment.
</item 31>

<item 32>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
The process also loops in the European Commission to help facilitate dialogue, assess issues and bring pressure to bear on unfair practices.

The CPC alert and mobilization process can be quicker than GDPR enforcement when it comes to forcing changes to unfair behaviors. Although it still typically
takes months, plural, for the network to coordinate and arrive at a position to press on a trader they believe is infringing the law.

The network also can’t impose fines itself. But if issues aren’t resolved through the dialogues and commitments process it shoots for, national consumer
protection authorities can still pursue enforcement at a local level. So if, at the end of the day, these consumer advocacy groups aren’t happy with whatever
the process of pressing Meta for changes will have achieved they can still press complaints to national authorities to urge them to take enforcement action (and
those CPAs have the ability to impose penalties of up to 4% of global annual turnover).
</item 32>

<item 33>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Aside from forcing platforms to allow some forms of content that would otherwise be disallowed, these laws also seek to force social media companies to provide
users with individualized explanations when their content is removed or restricted. Because this process is largely conducted algorithmically now — generally
with light human intervention or oversight — social media companies might need to reimagine their content moderation systems or hire way more humans to respond
to these incidents. Those kind of adjustments would likely be costly and difficult to scale.

“As you can imagine, that can get kind of onerous when you’re taking down millions and millions of pieces of content a day, and much of that activity of the
vast majority of it is currently being done automatically,” Barrett said. “The idea that a human being would have to go back and explain each time something
came down would be quite a challenge.”
</item 33>

<item 34>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
But it has yet to communicate a conclusion. In the meanwhile, Meta maintains that the model it has devised for obtaining users’ consent to its ads processing
complies with the General Data Protection Regulation (GDPR). (Although the adtech giant also said that when it was claiming performance of a contract and then
legitimate interests for the processing — both of which were subsequently found to be incompatible with the GDPR.)

The ‘pay or okay’ model Meta is seeking to impose on EU users wasn’t actually its invention; it was ‘pioneered’ in Austria, by the daily newspaper Der Standard
— after which copycat cookie paywalls quickly sprung up on a raft of news publishers in Germany and elsewhere in the EU.

noyb has been challenging this ‘pay or okay’ approach to GDPR consent since 2021— filing complaints with a number of data protection authorities, arguing the
model forces newspaper readers to “buy back their own data at exorbitant prices”.
</item 34>

<item 35>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
The lawsuit argues that “Meta does not obtain—or even attempt to obtain—verifiable parental consent before collecting the personal information of children on
Instagram and Facebook… But Meta’s own records reveal that it has actual knowledge that Instagram and Facebook target and successfully enroll children as
users.”

Essentially, while the problem of identifying kids’ accounts created in violation of platform rules is certainly a difficult one, Meta allegedly opted to turn a
blind eye for years rather than enact more stringent rules that would necessarily impact user numbers.

Meta, for its part, said in statements that the suit “mischaracterizes our work using selective quotes and cherry-picked documents,” and that “we have measures
in place to remove these [i.e. under-13] accounts when we identify them. However, verifying the age of people online is a complex industry challenge.”

Here are a few of the most striking parts of the suit.
</item 35>

<item 36>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
“We’re continuing to explore how to provide teens with a useful and responsible ad experience given this evolving regulatory landscape,” it writes on that.
</item 36>

<item 37>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 37>

<item 38>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
That’s one takeaway, at least, from the recent shutdown of Pebble (formerly T2), a Twitter clone that put trust and safety at the forefront of its development
process. The app never gained more than 20,000 users, and the founders now believe that its approach to moderation may have been correct, but it wasn’t a growth
driver. In other words, people may say they want to feel safer when participating online, but clearly, that isn’t their only need.

Mozilla’s instance, still in private testing, has a small team of moderators that will grow when the instance launches to the public in the coming months.

“It’s not just a commitment, it’s beyond foundational for us — that is the thing that is going to be true about this instance,” O’Hara says of Mozilla’s plans
to heavily moderate its space. Though Mozilla will also rely on technology to do some of the work, it wants to make sure the human moderators, “feel protected,
supported and respected for their work,” she adds.
</item 38>

<item 39>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“And so I think that is what is the most problematic right now about that particular rule, and the way that it’s enforced.”

Over the 13 days leading up to the ceasefire between Hamas and Israel, 7amleh documented more than 500 reports of Palestinian “digital rights violations,”
including the removal and restriction of content, hashtags and accounts related to the conflict.

Meta blamed some of the instances of perceived censorship to technical issues, like one that prevented users in Palestine and Colombia from posting Instagram
Stories. It attributed others to human error, like blocking the hashtag for Al-Aqsa Mosque, the holy site where Israeli police clashed with Ramadan worshippers,
because it was mistaken for a terrorist organization. The company also blocked journalists in Gaza from WhatsApp without explanation.

The same month, a group of Facebook employees filed internal complaints accusing the company of bias against Arab and Muslim users. In internal posts obtained
by BuzzFeed News, an employee attributed the bias to “years and years of implementing policies that just don’t scale globally.”
</item 39>

<item 40>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Meta also says it’s prioritizing checks on livestreaming functions on Facebook and Instagram — including watching for any attempts by Hamas to use the tools to
broadcast footage of captured Israelis or other hostages.

In a particularly disturbing report on Israeli media this week, which was widely recirculated on social media, a girl recounted how she and her family had
learnt about the death of her grandmother after Hamas militants had uploaded a video of her dead body to Facebook, apparently using her own mobile phone to post
the graphic content to the dead woman’s Facebook page.

“We recognize that the immediacy of Live brings unique challenges, so we have restrictions in place on the use of Live for people who have previously violated
certain policies. We’re prioritizing livestream reports related to this crisis, above and beyond our existing prioritization of Live videos,” Meta wrote,
highlighting measure it took in the wake of the 2019 Christchurch attacks in New Zealand when a single shooter livestreamed a killing spree that targeted two
mosques on Facebook.
</item 40>

<item 41>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
In response to the escalating violence, Meta said that it is closely monitoring its platforms for violations and may inadvertently flag certain content, but it
never intends to “suppress a particular community or point of view.” Content praising or supporting Hamas, which governs Gaza and is designated as a terrorist
organization by the United States and the European Union, is expressly forbidden on Meta’s platforms.

As the humanitarian crisis in Gaza grows more dire, many social media users suspect Instagram of censoring content about the besieged Palestinian territory,
even if that content doesn’t support Hamas. Users have also complained that they’ve been harassed and reported for posting content about Palestine, regardless
of whether or not it violates Meta’s policies. Jabr, for example, suspects that Instagram restricted her for 24 hours because other users reported her Palestine
travel videos. Most recently, Instagram users accused Meta of “shadowbanning” their Stories about Palestine.
</item 41>

<item 42>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
For example, will the survey go to under 13 year olds? Since everyone needs to be at least 13 years old before they create an account, we want to be careful
about sharing findings that come back and point to under 13 year olds being bullied on the platform.”

In 2021, another, studying “child-adult sexual-related content/behavior/interactions” (!) said she was “not includ[ing] younger kids (10-12 yos) in this
research” even though there “are definitely kids this age on IG,” because she was “concerned about risks of disclosure since they aren’t supposed to be on IG at
all.”

Also in 2021, Meta instructed a third-party research company conducting a survey of preteens to remove any information indicating a survey subject was on
Instagram, so the “company won’t be made aware of under 13.”
</item 42>

<item 43>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” he warns. “We have, from qualified sources, reports about potentially illegal content circulating on your service despite flags from relevant
authorities.”

He also takes issue with a change X made last night to its public interest policy, under which it judges newsworthiness (i.e., for deciding whether posts that
infringe its rules may nonetheless remain on the site) — but which Breton says have left “many European users uncertain” (i.e., about how X is applying its own
rules).

Again, this is a problem because the DSA requires platforms to be clear and transparent about their rules and how they apply them. “This is particularly
relevant when it comes to violent and terrorist content that appears to circulate on your platform,” Breton continues in another pointed warning.
</item 43>

<item 44>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
The Commission designated 19 VLOPs back in April — including Meta owned Facebook and Instagram.

The designation puts obligations on VLOPs to respond diligently to reports of illegal content, as well as clearly communicate their T&Cs to users and properly
enforce their terms. But it also ranges more widely — requiring these larger platforms to take steps to identify and mitigate systemic risks such as
disinformation.

The regulation also contains a “crisis response” mechanism which the Commission may adopt on VLOPs in situations where use of their platforms could contribute
to serious threats such as war. Although this aspect of the regulation won’t start operating until early next year (Feburary 17) as it requires an advisory
body, called the European Board for Digital Services, to be operational so it can recommend the Commission to adopt a crisis response mechanism (and the set up
of the Board is waiting on Member States to establish national Digital Services Coordinators, aka the bodies that will oversee DSA compliance for non-VLOPs).
</item 44>

<item 45>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Penalties for failing to comply with the pan-EU regulation can reach as high as 6% of global annual turnover — which, in Meta’s case, could run to multiple
billions.

The social media giant is not alone in being warned by the bloc over content concerns attached to the Israel-Hamas war: Elon Musk’s X has been singled out for
even greater attention here — with the bloc issuing both an “urgent” warning earlier this week and following that with a formal request for information about
its compliance approach.

TikTok has also received a warning from the EU about DSA content risks related to the conflict.

This report was updated with details about the operational timeline for the DSA crisis response mechanism coming into operation
</item 45>

<item 46>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
“We have no moat, and neither does OpenAI,” the document noted.

The memo in question was in reference to a foundation language model initially leaked from Meta in March, and which gained a fair bit of steam in a short period
of time. This highlighted the power and scalability of a more open approach to AI development — it enables collaboration and experimentation on a level that’s
not so easy to replicate with closed models.

It’s worth noting here that despite Meta’s claims, its Llama-branded family of LLMs are not as “open source” as it would like people to believe. Yes, they are
available for both research and commercial use cases, but it forbids developers to use Llama for training other models, while app developers with more than 700
million monthly users must request a special license from Meta which it may grant based on its “sole discretion” — basically, anyone but Meta’s Big Tech
brethren can use Llama sans permission.
</item 46>

<item 47>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The governor also signed a bill to prevent social media companies from adding features designed to addict minors to their platforms. A judge in Arkansas
recently blocked a law that would have required parental permission for minors to create social media accounts.

Beyond Meta, Montana also became the first state to ban TikTok, claiming it was being used by the Chinese Communist Party using TikTok to spy on Americans. But
Meta’s threat, in terms of teen usage, isn’t one of a foreign power collecting private data on users, but rather risks to teen mental health, body comparison
issues and exposure to sexual predators. Montana could easily turn its eyes to Meta next, after the TikTok bill went through.

The overhead of managing apps to meet individual state regulations would be a headache for Meta, but Davis also argues it would lead to a lack of teen
protection, in some cases.

“U.S.
</item 47>

<item 48>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.
</item 48>

<item 49>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
So Meta’s updated offer to EU users is either hand over your privacy or hand over your hard earned cash.

“This is an unfair choice for users, which runs afoul of EU consumer law on several counts and must be stopped,” said the European Consumer Organisation (BEUC)
in a press release announcing the complaint will be filed with the network of consumer protection authorities (CPC) today.

BEUC has been joined in the complaint by 18 of its member organizations — a variety of consumer advocacy groups which are located in the following EU member
states: Bulgaria, Czech Republic, Denmark, France, Greece, Italy, Latvia, Lithuania, Netherlands, Norway, Poland, Slovakia, Slovenia, Spain and Sweden.

The groups are objecting both to how Meta has gone about implementing the “pay-or-consent model” — using what they assess as “unfair, deceptive and aggressive
practices” — and to the model itself, which they dub “illegal”.
</item 49>

<item 50>
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
</item 50>

<item 51>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Asked whether the issue for consumer protection authorities is the ‘pay or consent’ choice Meta is offering or how it’s gone about implementing it, BEUC’s
spokesman said the questions are hard to separate as they’re “closely interlinked”.

“Under consumer law, you need an informed and fair choice to purchase such a subscription. The first question is also dependent on compliance with data
protection law. If the practice infringes the GDPR, the fact that it infringes a law which aims to protect fundamental rights should in our opinion be
considered unfair and illegal under consumer law too,” he told us, adding: “In any case, the choice is designed in a way that is unfair, aggressive and
misleading.”

The European Commission itself has an additional oversight role on Meta directly as the company is also subject to the EU’s shiny new Digital Markets Act (DMA)
and Digital Services Act (DSA).
</item 51>

<item 52>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
In recent times, a raft of complaints to the CPC about TikTok led — just last year — to the video sharing social network pledging to improve user reporting and
disclosure requirements around ads/sponsored content; and to boost transparency around its digital coins and virtual gifts. Although BEUC was not ecstatic about
the outcome, saying significant concerns remained unaddressed.

Still, the CPC network may be able to extract some ‘quick win’ concessions from Meta — such as requiring it to amend how it presents the available ‘Hobson’s
choice’ to users. Meta could also potentially face pressure to lower the subscription fee to make it more more affordable for users to deny tracking. (Just
spitballing here but imagine if it were offering a choice of tracking ads vs paying €1 a year not to be tracked which wouldn’t look so evidently self-serving.)
</item 52>

<item 53>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Through persistence and by creating a sense of urgency, Meta pushes consumers into making a choice they might not want to take.

In addition, many consumers likely think that, by opting for the paid subscription as it is presented, they get a privacy-friendly option involving less
tracking and profiling. In fact, users are likely to continue to have their personal data collected and used, but for purposes other than ads.

Meta provides misleading and incomplete information to consumers which does not allow them to make an informed choice. Meta is misleading them by presenting the
choice as between a paying and a ‘free’ option, while the latter option is not ‘free’ because consumers pay Meta through the provision of their data, as past
court rulings have already declared.
</item 53>

<item 54>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Given the market power of Meta’s Facebook and Instagram services in the EU and the very strong network effects of social media platforms (since all your friends
are on Facebook and Instagram), consumers do not have a real choice because if they quit the services they would lose all their contacts and interactions built
over the years. The very high subscription fee for ‘ad-free’ services is also a deterrent for consumers, which means consumers do not have a real choice.

“The company’s approach also raises concerns regarding the GDPR,” Pachl further noted. And a spokesman for BEUC told us it might, at a later stage, file a
complaint about Meta’s data protection compliance with the relevant privacy authority, once it has completed its own assessment of the issues. Although he
emphasized it’s too early to say whether or not it will take that step.

Meta’s lead data protection authority in the EU, Ireland’s Data Protection Commission (DPC), has, for several months, been assessing its pay or consent offer.
</item 54>

<item 55>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Though he’s glad that access to the technology has broadened, that has come with many complications.

“I don’t like the idea that the internet I’m so nostalgic for was actually an ivory tower,” he said, but on the flip side, he said, we’re all exposed to more
people, in real-time, than we could have ever fathomed. And that can put people at odds with one another and with everyone’s different versions of the truth —
even if their truth is believing in something as far-fetched as the earth being flat, he said, pointing to one example that has a following on Reddit.

Meanwhile, asking platforms like Facebook, Twitter, or Reddit to be in charge of verifying truth and accuracy is a difficult task, he said. Ohanian also called
out TikTok as “an arm for the Chinese government” that’s really good at “feeding us what we want, when we want it” — in other words, not always a fountain of
truth, either.
</item 55>

<item 56>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 56>

<item 57>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Per Mekić, the Commission’s ad campaign, which apparently racked up millions of views on X, insinuated that opponents of the proposed legislation did not want
to protect children — messaging he dubbed “a form of emotional blackmail”.

The ads included what he suggested is a misleading claim that the majority of Europeans support the proposal — but which is based on a survey that highlighted
“only the benefits but not the drawbacks of the proposed legislation”. Other surveys, by research firms YouGov and Novus, that highlighted the drawbacks showed
“virtually no support” for the plan among the European population, his post also pointed out.
</item 57>

<item 58>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
A newly unredacted version of the multi-state lawsuit against Meta alleges a troubling pattern of deception and minimization in how the company handles kids
under 13 on its platforms. Internal documents appear to show that the company’s approach to this ostensibly forbidden demographic is far more laissez-faire than
it has publicly claimed.

The lawsuit, filed last month, alleges a wide spread of damaging practices at the company relating to the health and well-being of younger people using it. From
body image to bullying, privacy invasion to engagement maximization, all the purported evils of social media are laid at Meta’s door — perhaps rightly, but it
also gives the appearance of a lack of focus.

In one respect at least, however, the documentation obtained by the attorneys general of 42 states is quite specific, “and it is damning,” as AG Rob Bonta of
California put it. That is in paragraphs 642 through 835, which mostly document violations of the Children’s Online Privacy Protection Act, or COPPA.
</item 58>

<item 59>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 59>

<item 60>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Responding to the escalating violence earlier this month, Meta stated that it established a “special operations center” staffed with fluent Hebrew and Arabic
speakers to closely monitor and respond to violating content.

Despite Meta’s apparent efforts to diversify its language resources, Arabic is still disproportionately flagged as violating — like in the case of journalists
reporting news in multiple languages.

“The balance of power is very fixed, in reality, between Israelis and Palestinians,” Abukhater said. “And this is something that today is reflected heavily on
platforms like Meta, even though they have human rights teams releasing reports and trying to improve upon their policies. Whenever an escalation like the one
we’re experiencing now happens, things just go back to zero.”

And at times, Meta’s Arabic translations are completely inaccurate. This week, multiple Instagram users raised concerns over the platform mistranslating the
relatively common Arabic phrase “Alhamdulillah,” or “Praise be to God.”
</item 60>

<item 61>
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
</item 61>

<item 62>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
In an Instagram Story last year, supermodel and activist Bella Hadid, who is of Palestinian descent, alleged that Instagram “disabled” her from posting content
on her Story “pretty much only when it is Palestine based.” She said she gets “immediately shadowbanned” when she posts about Palestine, and her Story views
drop by “almost 1 million.”

As soon as Kehlani got super vocal and changed her profile picture to “i stand with Palestine” she got shadowbanned.

Just as they did to Bella Hadid. The propaganda is insane.

THEY TRY TO SILENCE US. DON’T STOP!!

FREE PALESTINE. 🇵🇸🇵🇸 pic.twitter.com/4lUvZsUbiA — Sabby Targaryen 🐉 (@whoisitbad) October 14, 2023

Meta blamed technical errors for the removal of posts about Palestine during the 2021 conflict.
</item 62>

<item 63>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
(And on that front, Norway’s DPA, which has a local ban order on Meta’s tracking ads, told us it’s concerned about the subscription plan, saying it has doubts
as to whether it would constitute valid consent.)

As well as needing to comply with the GDPR, which delineates the qualities necessary for consent to be legal (such as that it is specific, informed and freely
given), Meta is now also subject to the pan-EU Digital Services Act (DSA) — which also sets conditions on larger platforms when it comes to tracking and
profiling people for ads. So it won’t just be data protection authorities making a call on whether Meta’s subscription or tracking offer flies; the European
Commission is responsible for oversight of very large online platforms’ DSA compliance.

Meta has also been designated as a so-called gatekeeper, under the DSA’s sister regulation, the Digital Markets Act (DMA) — which also puts some limits on use
of people’s data for ads. The Commission is the sole enforcer of the DMA.
</item 63>

<item 64>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“Thanks to the First Amendment, Florida and Texas cannot force websites or social media apps to host hateful content, misinformation and spam, as their deeply
misguided laws would require,” Oregon Senator Ron Wyden, who co-authored Section 230, a law that protects social media companies’ content moderation decisions,
told TechCrunch. “… A ruling in favor of the Texas and Florida laws would create utter chaos, and make many sites worthless to regular users who want to watch a
funny video or see family photos.”

NetChoice President Steve DelBianco also warned that allowing the state laws to go into effect would unleash “a tidal wave of offensive content and hate speech
crashing onto users, creators, and advertisers” that would force Americans to wade through “racial epithets, aggressive homophobia, pornographic material,
beheadings, or other gruesome content” just to use social apps.
</item 64>

<item 65>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
In the latter case its social networks, Facebook and Instagram, were designated as very large online platforms (VLOPs) earlier this year. And, since late
August, they’ve been expected to be compliant with that digital rulebook.

Both pan-EU laws put restrictions on the use of personal data for advertising — explicitly requiring consent is obtained from users for such a purpose; and that
consent must be as easy to withhold as to affirm. So one issue the Commission — which is the sole enforcer of the DSA on VLOPs — might weigh in in the coming
months on is whether clicking accept vs digging out a credit card to pay a monthly charge are equivalently easy.

The regulation also contains provisions which are intended to combat unfair/deceptive design, such as targeting choice interfaces that make it “more difficult
or time-consuming” to pick one option over another.
</item 65>

<item 66>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 66>

<item 67>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
When reached for comment about these recent claims of shadowbanning, a representative for the company pointed TechCrunch to a Threads post by Stone.

“We identified a bug impacting all Stories that re-shared Reels and Feed posts, meaning they weren’t showing up properly in people’s Stories tray, leading to
significantly reduced reach,” Stone said. “This bug affected accounts equally around the globe and had nothing to do with the subject matter of the content —
and we fixed it as quickly as possible.”

But many are frustrated that Meta continues to disproportionately suppress Palestinian voices. Leen Al Saadi, a Palestinian journalist currently based in Jordan
and host of the podcast “Preserving Palestine,” said she is used to “constantly being censored.” Her Instagram account was restricted last year after she posted
a trailer for the podcast’s first episode, which discussed a documentary about Palestinian street art under occupation.

“Palestinians are currently undergoing two wars,” Al Saadi said. “The first is with their legal occupier.
</item 67>

<item 68>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
In screen recordings posted online, users found that if they included “Palestinian” and the corresponding flag emoji in their Instagram bio along with the
Arabic phrase, Instagram automatically translated their bio to “Palestinian terrorists – Praise be to Allah” or “Praise be to God, Palestinian terrorists are
fighting for their freedom.” When users removed “Palestinian” and the flag emoji, Instagram translated the Arabic phrase to “Thank God.” Instagram users
complained that the offensive mistranslation was active for hours before Meta appeared to correct it.

Shayaan Khan, a TikTok creator who posted a viral video about the mistranslation, told TechCrunch that Meta’s lack of cultural competence isn’t just offensive,
it’s dangerous. He said that the “glitch” can fuel Islamophobic and racist rhetoric, which has already been exacerbated by the war in Gaza. Khan pointed to the
fatal stabbing of Wadea Al-Fayoume, a Palestinian-American child whose death is being investigated as a hate crime.
</item 68>

<item 69>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
However the adtech giant’s interpretation of free consent with this ‘pay us or be tracked’ subscription proposal will, justifiably, leave privacy advocates
fuming — since the choice it’s offering here boils down to ‘pay us money; or pay us with your privacy’.

Per Meta’s blog post, the fee it plans to charge users to escape its tracking and targeting (i.e. the ad-free subscription) is €9.99/month on web or
€12.99/month on iOS or Android per linked Facebook and Instagram accounts in a user’s Accounts Center. After March 1, 2024 it also says an additional fee — of
€6/month on web and €8/month on iOS or Android — will apply for each additional account listed in a user’s Account Center.

So the cost for using Meta’s services without being tracked and profiled could quickly stack up for anyone with more than one account on Meta’s social networks.
</item 69>

<item 70>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.
</item 70>

<item 71>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
Meta is to offer an ad-free subscription version of Facebook and Instagram in the European Union, EEA (European Economic Area) and Switzerland, confirming the
core of a report in the WSJ earlier this month. The new ad-free subscription will be available from next month, per a Meta blog post.

The move follows years of privacy litigation, enforcements and court rulings in the EU — which have culminated in a situation where Meta can no longer claim a
contractual right (nor legitimate interest) to track and profile users for ad targeting. (Although, at the time of writing, it is still doing the latter —
meaning it is technically operating without a proper legal basis. But this summer Meta announced an intention to switch to consent.)

Under regional data protection law the only available basis left for Meta’s tracking and profiling ad business is to obtain freely given consent from users.
</item 71>

<item 72>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
They choose to do things that maximize the frequency of use of that platform by children, and the duration of that use. They could end all this today if they
wanted, they could easily keep those under 13 from accessing their platform. But they’re not.”

You can read the mostly unredacted complaint here.

(This story has been updated with a comment from Meta.)
</item 72>

<item 73>
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
</item 73>

<item 74>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
“U.S. states are passing a patchwork of different laws, many of which require teens (of varying ages) to get their parent’s approval to use certain apps, and
for everyone to verify their age to access them,” she wrote. “Teens move interchangeably between many websites and apps, and social media laws that hold
different platforms to different standards in different states will mean teens are inconsistently protected,” Davis said.

Meta’s proposed “better way” of requiring parents to manage a teen’s app downloads pushes the burden back to the app stores and to parents, who often today do
not monitor or manage their teens’ smartphone use. But while it’s true that the app stores and smartphone makers would know the ages of their users, requiring
parents to approve apps is the equivalent of overriding parental discretion and choice.

Other entertainment ratings — like those for TV, movies or video games — are meant to be guidelines for parents, not laws. Meta’s position that apps are somehow
different, then, is a fairly extreme position to take.
</item 74>

<item 75>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Days after a privacy complaint was lodged against Meta in the European Union over its latest controversial shift of legal basis claimed for processing people’s
data for ads, consumer groups across the region are filing their own complaints about what the tracking giant is up to.

A coalition of almost 20 consumer protection organizations is united in the view that Meta’s switch to railroading users into agreeing to being tracked and
profiled so it can keep profiting from microtargeting them is “unfair” and “illegal” — breaching EU consumer protection law “on several counts”.

Starting this month, EU users of Meta’s social networks, Facebook and Instagram, are being offered the ‘choice’ of agreeing to being tracked and profiled by the
behavioral ads business in order to continue/get free access to its products — or else they must pay it a monthly subscription (of at least €9.99pm) for an
ad-free version of its mainstream social networks.
</item 75>

<item 76>
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
</item 76>

<item 77>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Creators have tried to include an emoji of the Israeli flag or tag their posts and Stories with #istandwithisrael, even if they don’t support the Israeli
government, in hopes of gaming engagement.

Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.

“All we’re asking for is to give us the exact same rights,” she said. “We’re not asking for more. We’re literally just asking Meta, Instagram, every single
broadcast channel, every single media outlet, to just give us the respect that we deserve.”

Dominic-Madori Davis contributed to this story’s reporting.
</item 77>

<item 78>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
In a press release earlier this month after the WSJ reported Meta plans to charge users for their privacy, noyb’s founder and honorary chairman, Max Schrems,
wrote: “The CJEU said that the alternative to ads must be ‘necessary’ and the fee must be ‘appropriate’. I don’t think €160 a year is what they had in mind.
These six words are also an ‘obiter dictum’, a non-binding element that went beyond the core case before the CJEU. For Meta this is not the most stable case law
and we will clearly fight against such an approach.”

Contacted for a response to the development, Meta’s lead regulator in the EU for the General Data Protection Regulation (GDPR), Ireland’s Data Protection
Commission (DPC) sent us a statement.
</item 78>

<item 79>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
“We’re also aware of Hamas’ threats to broadcast footage of the hostages and we’re taking these threats extremely seriously. Our teams are monitoring this
closely, and would swiftly remove any such content (and the accounts behind it), banking the content in our systems to prevent copies being re-shared,” it
added.

Other steps taken by Meta to respond to the situation in Israel and Gaza include making it less likely that its systems will actively recommend potentially
violating or borderline content and to reduce the visibility of potentially offensive comments; and applying hashtag blocking to render certain terms related to
the conflict non-searchable on its platforms. Its blog post does not specify which hashtags Meta is blocking in relation to the Israel-Hamas war.

Meta’s blog post also says it established a special operations center staffed with experts, including Arabic and Hebrew speakers, to dial up its ability to
quickly respond to content report.

It also says it’s taking feedback from local partners (such as NGOs) on emerging risks — and claiming to be “moving quickly to address them”.
</item 79>

<item 80>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
These included findings made by the Court of Justice of the European Union when delivering judgment on July 4, 2023 in a case in which the Court examined the
legal bases on which Meta’s processing of users’ data for behavioural advertising purposes is grounded.

“Acting in consultation with its fellow European supervisory authorities, the DPC has been engaged in a detailed regulatory assessment of the consent-based
model since it was first proposed by Meta in July. That exercise is being led by the DPC, reflecting its position as Lead Supervisory Authority for Facebook and
Instagram in Europe. The exercise has not yet concluded, and no findings have been made to date. It is due to be completed shortly, at which point the DPC will
notify Meta if it considers that the manner in which its new user offerings are to be implemented is compatible with Meta’s obligations under GDPR.”

So Meta’s move to offer users a subscription or tracking hasn’t been signed off by data protection authorities as yet — and it’s clear that further regulatory
interventions could follow.
</item 80>

<item 81>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.
</item 81>

<item 82>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
One stated goal, according to the lawsuit, was specifically to “grow [Monthly Active People], [Daily Active People] and time spent among U13 kids.”

It’s important to note here that while Meta does not permit accounts to be run by people under 13, there are plenty of ways it can lawfully and safely engage
with that demographic. Some kids just want to watch videos from SpongeBob Official, and that’s fine. However, Meta must verify parental consent and the ways it
can collect and use their data is limited.

But the redactions suggest these under-13 users are not of the lawfully and safely engaged type. Reports of underage accounts are reported to be automatically
ignored, and Meta “continues collecting the child’s personal information if there are no photos associated with the account.” Of 402,000 reports of accounts
owned by users under 13 in 2021, fewer than 164,000 were disabled.
</item 82>

<item 83>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
The Commission is the sole enforcer of the DMA.

Meta is already in the Commission’s sights over its approach to the DSA — with the EU’s executive, in recent days, requesting more information from the tech
giant vis-à-vis its approach to content threats flowing from the Israel-Hamas war and about election security issues. But it remains to be seen whether the EU
will apply the same close scrutiny to Meta’s ad tracking offer.

In its blog post Meta claims the choice to offer people to pay it for their privacy or else agree to being tracked “balances the requirements of European
regulators while giving users choice and allowing Meta to continue serving all people in the EU, EEA and Switzerland”. But, well, it would say that.

The subscription is also only being offered to people aged 18 and up — which raises questions over how it will comply with requirements in the DSA and DMA not
to processing children’s data for ad targeting.
</item 83>

<item 84>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
I think even swifter than we realize.” The war in the Middle East may even be the tipping point, he theorized, as it’s now “abundantly clear” that we need
something better to figure out the world than the platforms we have today.

The founder, as you may recall, notably resigned from Reddit’s board in 2020 over differences of opinion on moderation and wanting to crack down on communities
about violence and hate. It was a moment where he seemed to be reckoning with what his platform had wrought. At the time, he asked for his board seat to be
replaced by a Black board member. The company agreed, appointing its first Black board member with Y Combinator partner Michael Seibel.

Reddit also later banned several toxic communities, including r/donaldtrup following violence at the U.S. Capitol, and Ohanian says that the site has “gotten
demonstrably better,” since, as well as better as a business.
</item 84>

<item 85>
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
</item 85>

<item 86>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Users report that Instagram has flagged comments containing the emoji as “potentially offensive,” hiding the comment.

Meta initially did not respond to follow-up requests for comment. When asked for comment, Stone pointed TechCrunch to Meta’s updated newsroom post, which did
not directly address issues related to comments containing the Palestinian flag.

The organization has also received countless reports of Meta flagging and restricting Arabic content, even if it’s posted by news outlets. Jalal Abukhater,
7amleh’s advocacy manager, said that the organization has documented multiple cases of journalists on Instagram reporting the same news in Arabic, Hebrew and
English, but only getting flagged for their Arabic content.

“It’s literally journalistic content, but the same wording in Hebrew and English does not get restricted,” Abukhater said. “As if there’s better moderation for
those languages, and more careless moderation for Arabic content.”
</item 86>

<item 87>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta did not respond to TechCrunch’s request for comment about the mistranslation. Abukhater said that Meta told 7amleh that a “bug” caused the mistranslation.
In a statement to 404 Media, a Meta spokesperson said that the issue had been fixed.

“We fixed a problem that briefly caused inappropriate Arabic translations in some of our products,” the statement said, “We sincerely apologize that this
happened.”

As the war continues, social media users have tried to find ways around the alleged shadowbanning on Instagram. Supposed loopholes include misspelling certain
words, like “p@lestine” instead of “Palestine,” in hopes of bypassing any content filters. Users also share information about Gaza in text superimposed over
unrelated images, like a cat photo, so it won’t be flagged as graphic or violent content.
</item 87>

<item 88>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
Ofcom recommends these platforms implement policies like not allowing strangers to send direct messages to children, using hash matching to detect and remove
CSAM, maintaining content and search moderation teams, and offering ways for users to report harmful content.

Large tech platforms already follow many of these practices, but Ofcom hopes to see them implemented more consistently. “We think they represent best practice
of what’s out there, but it’s not necessarily applied across the board,” Whitehead says. “Some firms are applying it sporadically but not necessarily
systematically, and so we think there is a great benefit for a more wholesale, widespread adoption.”

There’s also one big outlier: the platform known as X (formerly Twitter). The UK’s efforts with the legislation long predate Elon Musk’s acquisition of Twitter,
but it was passed as he fired large swaths of its trust and safety teams and presided over a loosening of moderation standards, which could put X at odds with
regulators.
</item 88>

<item 89>
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
</item 89>

<item 90>
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
</item 90>

<item 91>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Some DPAs appear to have been sympathetic to local newspapers’ use of cookie walls, seeing it as a way to support the production of journalism. However when it
comes to Meta, that argument evaporates as it’s definitely not in the journalism business. Moreover the adtech giant doesn’t even need to produce content to
pump around its social networks; it gets all that filler for free from the self-same users it’s now demanding pay a fee if they want to use its services without
being tracked and profiled for behavioral advertising. Which, well, makes Meta’s ‘pay or okay’ model feel like even more of a rip off.

Back in April, a decision by Austria’s DPA on a noyb complaint about cookie paywalls said users must have the ability to say yes or no to specific data
operations — meaning blanket consent is not an option.
</item 91>

<item 92>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Confirmed breaches of the online governance regime can face a range of major sanctions, including fines of up to 6% of global annual turnover.

The EU can also apply interim measures where it believes there’s a risk of serious harm for users and may even seek to have access to infringing services
temporarily blocked. Although its announcement today is careful to caveat the development by affirming no conclusions have yet been reached.

There is also no confirmed timeline for the probe to conclude. But the fact of it being opened means other tools are put within reach of regulators and could be
applied if the EU sees the need to act quickly — such as the aforementioned interim measures. So the Commission’s official scrutiny of X could have real world
implications for how the platform operates sooner rather than later.
</item 92>

<item 93>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
LeCun et al. disagree.

“Yes, openly available models come with risks and vulnerabilities — AI models can be abused by malicious actors or deployed by ill-equipped developers,” the
letter acknowledged. “However, we have seen time and time again that the same holds true for proprietary technologies — and that increasing public access and
scrutiny makes technology safer, not more dangerous. The idea that tight and proprietary control of foundational AI models is the only path to protecting us
from society-scale harm is naive at best, dangerous at worst.”

On a personal level, LeCun has accused the big-name AI players of trying to secure “regulatory capture of the AI industry” by lobbying against open AI R&D. And
on a company level, Meta is doing all it can to encourage collaboration and “openness,” recently partnering with Hugging Face to launch a new startup
accelerator designed to spur adoption of open source AI models.

But OpenAI was — up until last week, at least — still the AI darling everyone wanted to dance with.
</item 93>

<item 94>
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
</item 94>

<item 95>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
Here are a few of the most striking parts of the suit. While some of these allegations relate to practices from years ago, bear in mind that Meta (then
Facebook) has been publicly saying it doesn’t allow kids on the platform, and diligently worked to detect and expel them, for a decade.

Meta has internally tracked and documented under-13s, or U13s, in its audience breakdowns for years, as charts in the filing show. In 2018, for instance, it
noted that 20% of 12-year-olds on Instagram used it daily. And this was not in a presentation about how to remove them — it is relating to market penetration.
The other chart shows Meta’s “knowledge that 20-60% of 11- to 13-year-old users in particular birth cohorts had actively used Instagram on at least a monthly
basis.”

It’s hard to square this with the public position that users this age are not welcome. And it isn’t because leadership wasn’t aware.
</item 95>

<item 96>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“This was also a global issue that was fixed within a few hours. We understand people rely on these tools and we’re sorry to anyone who felt the impact of these
issues,” the post stated.

It’s unclear if this “bug” impacted accounts posting content unrelated to Gaza. Instagram users have posted screenshots showing that Stories about Palestine
have received significantly fewer views than other Stories posted on the same day, and allege that their view counts went back up when they posted content
unrelated to the conflict.

ofc IG’s hiding my stories about Palestine, y’all see the difference? lmaoaoaoao pic.twitter.com/1iYIIKBhtN — ༒︎ (@arxbprince) October 15, 2023

A user based in Egypt, who asked to stay anonymous for fear of harassment, said her posts usually get around 300 views, but when she started posting
pro-Palestine content after the Hamas attack earlier this month, her stories would only get one to two views.
</item 96>

<item 97>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices Instagram users suspect Meta of shadowbanning content about
Palestine. The platform has a history of inherent bias.

Earlier this year, Palestinian-American filmmaker Khitam Jabr posted a handful of Reels about her family’s trip to the West Bank. In the short travel vlogs,
Jabr shared snippets of Palestinian culture, from eating decadent meals to dancing at her niece’s wedding.

“I hadn’t been in a decade, so it’s just like, life abroad,” Jabr told TechCrunch.

“I would get [anti-Palestine] comments,” she recalled. “And I couldn’t respond [to them] or use my account for 24 hours. I wasn’t even posting anything about
the occupation. But fast forward to now and the same shit’s happening.”

In the aftermath of Hamas’ attack on Israelis, Israel’s retaliatory airstrikes and total blockade — cutting access to electricity, water and vital supplies —
have devastated Gaza.
</item 97>

<item 98>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
“In the three days following October 7, we removed or marked as disturbing more than 795,000 pieces of content for violating these policies in Hebrew and
Arabic,” it wrote. “As compared to the two months prior, in the three days following October 7, we have removed seven times as many pieces of content on a daily
basis for violating our Dangerous Organizations and Individuals policy in Hebrew and Arabic alone.”

In light of the dialled up attention on and concern about the situation, Meta says it’s possible non-violating content may be removed “in error”.

“To mitigate this, for some violations we are temporarily removing content without strikes, meaning these content removals won’t cause accounts to be disabled,”
it notes. “We also continue to provide tools for users to appeal our decisions if they think we made a mistake.”

Compliance with the bloc’s Digital Services Act (DSA) kicked in for Meta in August as the owner of a so-called very large online platform (VLOP).
</item 98>

<item 99>
People are turning to Snap Map for firsthand perspectives from Gaza (TechCrunch, 2023-11-03T17:24:18+00:00)
The submission guidelines say that the company looks for “Snaps that provide a window into everyday life.”

Users are also turning to Snapchat for information about Gaza as accusations of shadowbanning Palestinian content fuel distrust in platforms like Instagram and
Facebook. Meta’s policies aren’t dissimilar to Snap’s Community Guidelines, which prohibit pornographic content, content that advocates or advances terrorism
and content that portrays gratuitous or graphic violence. Allegations of censorship against Snapchat aren’t as frequent as they are against Meta’s platforms,
but many users are still urging others to screen record Stories posted from Gaza in case they are taken down.

Some are also concerned that internet access in Gaza, which was partially restored after Israel imposed a communications blackout, will be cut again. TikTok
creator divergentredhed posted that she watched the blackout happen in real time on Snapchat.

“Right there in Gaza it was like a hotspot of Snapchat Stories. It was yellow and green and red … all of it just went dark. The entire Gaza Strip went dark
before my very eyes,” she said.
</item 99>

<item 100>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.
</item 100>
=== END ITEM id=833791475b7b01aa ===

=== ITEM id=1d4b086e4154f4f7 ===
Gold sentences and facts:
Sentence 0: For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
  Fact 0.0: The victory has the Swans challenging for a top-eight spot.
  Fact 0.1: The Swans are challenging for a top-eight spot alongside the likes of Gold Coast.
  Fact 0.2: The Swans are challenging for a top-eight spot alongside the likes of Geelong.
  Fact 0.3: The Swans are challenging for a top-eight spot alongside the likes of Collingwood.
  Fact 0.4: The Swans are challenging for a top-eight spot alongside the likes of St Kilda.
  Fact 0.5: The Swans are challenging for a top-eight spot alongside the likes of Carlton.
Sentence 1: With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot.
  Fact 1.0: Sydney is in ninth place.
  Fact 1.1: St Kilda is in eighth place.
  Fact 1.2: Only 0.4 of a percentage point separates Sydney in ninth and St Kilda in eighth.
  Fact 1.3: The win assured the Swans of retaking eighth spot.
Sentence 2: “Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.
  Fact 2.0: The speaker said that her team has put a lot of work into sourcing some wonderful food offerings.
  Fact 2.1: The speaker said that her team has put a lot of work into sourcing entertainment for kids.
  Fact 2.2: The speaker said that her team has put a lot of work into sourcing activities for kids.
  Fact 2.3: The speaker said that her team has had a local focus in promoting its games.

Retrieved items:
<item 1>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 1>

<item 2>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Sit Him (Lineup Decisions) Projections powered by Sportsline Cooper Kupp WR LAR L.A. Rams • #10

Age: 30 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs CLE LAR -3.5 O/U 40 OPP VS WR 14th PROJ PTS 13.8 WR RNK 25th YTD Stats REC 27 TAR 51 REYDS 393 TD 1
FPTS/G 10.6 You're never supposed to sit your studs, but what if your stud is playing through something that made him look like a dud last week? That was Kupp,
who in Week 12 ran a ton of short routes and couldn't separate and didn't have much sudden movement. If he's going to come through this week it'll likely come
on a ton of short targets and/or some targets inside the 10-yard line.
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.
</item 3>

<item 4>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 4>

<item 5>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Week 11 of the 2023 NFL season has provided plenty of drama, from the Bears hanging with the Lions to the Giants getting a rebound from emergency quarterback
Tommy DeVito. But the latest round of regular-season action has also brought another wave of unfortunate injuries. Here's a roundup of the latest big names to
go down around the league:

QB

C.J. Stroud (Texans) left for a single play against the Cardinals after absorbing a hard hit, with backup Davis Mills entering in relief. Stroud did not receive
an injury designation, however, and returned after a quick check-in from trainers.

Geno Smith (Seahawks) suffered an injury to his throwing arm after guiding a 16-7 lead over the Rams, showing frustration while trying to stretch the arm after
a hit from Aaron Donald. Seattle quickly ruled him questionable to return with an elbow issue, and backup Drew Lock took over under center with 18 seconds left
in the third quarter.
</item 5>

<item 6>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 9>

<item 10>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 10>
=== END ITEM id=1d4b086e4154f4f7 ===
