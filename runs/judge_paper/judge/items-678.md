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

=== ITEM id=d5cdcdbefc952e9e ===
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
</item 1>

<item 2>
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
</item 2>

<item 3>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
“We’re continuing to explore how to provide teens with a useful and responsible ad experience given this evolving regulatory landscape,” it writes on that.
</item 3>

<item 4>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
Here are a few of the most striking parts of the suit. While some of these allegations relate to practices from years ago, bear in mind that Meta (then
Facebook) has been publicly saying it doesn’t allow kids on the platform, and diligently worked to detect and expel them, for a decade.

Meta has internally tracked and documented under-13s, or U13s, in its audience breakdowns for years, as charts in the filing show. In 2018, for instance, it
noted that 20% of 12-year-olds on Instagram used it daily. And this was not in a presentation about how to remove them — it is relating to market penetration.
The other chart shows Meta’s “knowledge that 20-60% of 11- to 13-year-old users in particular birth cohorts had actively used Instagram on at least a monthly
basis.”

It’s hard to square this with the public position that users this age are not welcome. And it isn’t because leadership wasn’t aware.
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
In an Instagram Story last year, supermodel and activist Bella Hadid, who is of Palestinian descent, alleged that Instagram “disabled” her from posting content
on her Story “pretty much only when it is Palestine based.” She said she gets “immediately shadowbanned” when she posts about Palestine, and her Story views
drop by “almost 1 million.”

As soon as Kehlani got super vocal and changed her profile picture to “i stand with Palestine” she got shadowbanned.

Just as they did to Bella Hadid. The propaganda is insane.

THEY TRY TO SILENCE US. DON’T STOP!!

FREE PALESTINE. 🇵🇸🇵🇸 pic.twitter.com/4lUvZsUbiA — Sabby Targaryen 🐉 (@whoisitbad) October 14, 2023

Meta blamed technical errors for the removal of posts about Palestine during the 2021 conflict.
</item 7>

<item 8>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
The lawsuit argues that “Meta does not obtain—or even attempt to obtain—verifiable parental consent before collecting the personal information of children on
Instagram and Facebook… But Meta’s own records reveal that it has actual knowledge that Instagram and Facebook target and successfully enroll children as
users.”

Essentially, while the problem of identifying kids’ accounts created in violation of platform rules is certainly a difficult one, Meta allegedly opted to turn a
blind eye for years rather than enact more stringent rules that would necessarily impact user numbers.

Meta, for its part, said in statements that the suit “mischaracterizes our work using selective quotes and cherry-picked documents,” and that “we have measures
in place to remove these [i.e. under-13] accounts when we identify them. However, verifying the age of people online is a complex industry challenge.”

Here are a few of the most striking parts of the suit.
</item 8>

<item 9>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
Later that year, external researchers provided Meta with information that “of children ages 9-12, 45% used Facebook and 40% used Instagram daily.”

During an internal 2021 study on youth in social media described in the suit, they first asked parents if their kids are on Meta platforms and removed them from
the study if so. But one researcher asked, “What happens to kids who slip through the screener and then say they are on IG during the interviews?” Instagram
Head of Public Policy Karina Newton responded, “we’re not collecting user names right?” In other words, what happens is nothing.

As the lawsuit puts it:

Even when Meta learns of specific children on Instagram through interviews with the children, Meta takes the position that it still lacks actual knowledge of
that it is collecting personal information from an under-13 user because it does not collect user names while conducting these interviews.
</item 9>

<item 10>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“And so I think that is what is the most problematic right now about that particular rule, and the way that it’s enforced.”

Over the 13 days leading up to the ceasefire between Hamas and Israel, 7amleh documented more than 500 reports of Palestinian “digital rights violations,”
including the removal and restriction of content, hashtags and accounts related to the conflict.

Meta blamed some of the instances of perceived censorship to technical issues, like one that prevented users in Palestine and Colombia from posting Instagram
Stories. It attributed others to human error, like blocking the hashtag for Al-Aqsa Mosque, the holy site where Israeli police clashed with Ramadan worshippers,
because it was mistaken for a terrorist organization. The company also blocked journalists in Gaza from WhatsApp without explanation.

The same month, a group of Facebook employees filed internal complaints accusing the company of bias against Arab and Muslim users. In internal posts obtained
by BuzzFeed News, an employee attributed the bias to “years and years of implementing policies that just don’t scale globally.”
</item 10>

<item 11>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
It’s difficult to extrapolate from the 2015 and 2020 numbers to today’s (which, as we have seen from the evidence presented here, will almost certainly not be
the whole story), but Bonta noted that the large figures are presented for impact, not as legal justification.

“The basic premise remains that their social media platforms are used by millions of children under 13. Whether it’s 30 percent, or 20 or 10 percent… any child,
it’s illegal,” he said. “If they were doing it at any time, it violated the law at that time. And we are not confident that they have changed their ways.”

An internal presentation called “2017 Teens Strategic Focus” appears to specifically target kids under 13, noting that children use tablets as early as 3 or 4,
and “Social identity is an Unmet need Ages 5-11.”
</item 11>

<item 12>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Users report that Instagram has flagged comments containing the emoji as “potentially offensive,” hiding the comment.

Meta initially did not respond to follow-up requests for comment. When asked for comment, Stone pointed TechCrunch to Meta’s updated newsroom post, which did
not directly address issues related to comments containing the Palestinian flag.

The organization has also received countless reports of Meta flagging and restricting Arabic content, even if it’s posted by news outlets. Jalal Abukhater,
7amleh’s advocacy manager, said that the organization has documented multiple cases of journalists on Instagram reporting the same news in Arabic, Hebrew and
English, but only getting flagged for their Arabic content.

“It’s literally journalistic content, but the same wording in Hebrew and English does not get restricted,” Abukhater said. “As if there’s better moderation for
those languages, and more careless moderation for Arabic content.”
</item 12>

<item 13>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Penalties for failing to comply with the pan-EU regulation can reach as high as 6% of global annual turnover — which, in Meta’s case, could run to multiple
billions.

The social media giant is not alone in being warned by the bloc over content concerns attached to the Israel-Hamas war: Elon Musk’s X has been singled out for
even greater attention here — with the bloc issuing both an “urgent” warning earlier this week and following that with a formal request for information about
its compliance approach.

TikTok has also received a warning from the EU about DSA content risks related to the conflict.

This report was updated with details about the operational timeline for the DSA crisis response mechanism coming into operation
</item 13>

<item 14>
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
</item 14>

<item 15>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
The Commission designated 19 VLOPs back in April — including Meta owned Facebook and Instagram.

The designation puts obligations on VLOPs to respond diligently to reports of illegal content, as well as clearly communicate their T&Cs to users and properly
enforce their terms. But it also ranges more widely — requiring these larger platforms to take steps to identify and mitigate systemic risks such as
disinformation.

The regulation also contains a “crisis response” mechanism which the Commission may adopt on VLOPs in situations where use of their platforms could contribute
to serious threats such as war. Although this aspect of the regulation won’t start operating until early next year (Feburary 17) as it requires an advisory
body, called the European Board for Digital Services, to be operational so it can recommend the Commission to adopt a crisis response mechanism (and the set up
of the Board is waiting on Member States to establish national Digital Services Coordinators, aka the bodies that will oversee DSA compliance for non-VLOPs).
</item 15>

<item 16>
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
</item 16>

<item 17>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Creators have tried to include an emoji of the Israeli flag or tag their posts and Stories with #istandwithisrael, even if they don’t support the Israeli
government, in hopes of gaming engagement.

Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.

“All we’re asking for is to give us the exact same rights,” she said. “We’re not asking for more. We’re literally just asking Meta, Instagram, every single
broadcast channel, every single media outlet, to just give us the respect that we deserve.”

Dominic-Madori Davis contributed to this story’s reporting.
</item 17>

<item 18>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.
</item 18>

<item 19>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
“In the three days following October 7, we removed or marked as disturbing more than 795,000 pieces of content for violating these policies in Hebrew and
Arabic,” it wrote. “As compared to the two months prior, in the three days following October 7, we have removed seven times as many pieces of content on a daily
basis for violating our Dangerous Organizations and Individuals policy in Hebrew and Arabic alone.”

In light of the dialled up attention on and concern about the situation, Meta says it’s possible non-violating content may be removed “in error”.

“To mitigate this, for some violations we are temporarily removing content without strikes, meaning these content removals won’t cause accounts to be disabled,”
it notes. “We also continue to provide tools for users to appeal our decisions if they think we made a mistake.”

Compliance with the bloc’s Digital Services Act (DSA) kicked in for Meta in August as the owner of a so-called very large online platform (VLOP).
</item 19>

<item 20>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
In response to the escalating violence, Meta said that it is closely monitoring its platforms for violations and may inadvertently flag certain content, but it
never intends to “suppress a particular community or point of view.” Content praising or supporting Hamas, which governs Gaza and is designated as a terrorist
organization by the United States and the European Union, is expressly forbidden on Meta’s platforms.

As the humanitarian crisis in Gaza grows more dire, many social media users suspect Instagram of censoring content about the besieged Palestinian territory,
even if that content doesn’t support Hamas. Users have also complained that they’ve been harassed and reported for posting content about Palestine, regardless
of whether or not it violates Meta’s policies. Jabr, for example, suspects that Instagram restricted her for 24 hours because other users reported her Palestine
travel videos. Most recently, Instagram users accused Meta of “shadowbanning” their Stories about Palestine.
</item 20>

<item 21>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
The Commission is the sole enforcer of the DMA.

Meta is already in the Commission’s sights over its approach to the DSA — with the EU’s executive, in recent days, requesting more information from the tech
giant vis-à-vis its approach to content threats flowing from the Israel-Hamas war and about election security issues. But it remains to be seen whether the EU
will apply the same close scrutiny to Meta’s ad tracking offer.

In its blog post Meta claims the choice to offer people to pay it for their privacy or else agree to being tracked “balances the requirements of European
regulators while giving users choice and allowing Meta to continue serving all people in the EU, EEA and Switzerland”. But, well, it would say that.

The subscription is also only being offered to people aged 18 and up — which raises questions over how it will comply with requirements in the DSA and DMA not
to processing children’s data for ad targeting.
</item 21>

<item 22>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The governor also signed a bill to prevent social media companies from adding features designed to addict minors to their platforms. A judge in Arkansas
recently blocked a law that would have required parental permission for minors to create social media accounts.

Beyond Meta, Montana also became the first state to ban TikTok, claiming it was being used by the Chinese Communist Party using TikTok to spy on Americans. But
Meta’s threat, in terms of teen usage, isn’t one of a foreign power collecting private data on users, but rather risks to teen mental health, body comparison
issues and exposure to sexual predators. Montana could easily turn its eyes to Meta next, after the TikTok bill went through.

The overhead of managing apps to meet individual state regulations would be a headache for Meta, but Davis also argues it would lead to a lack of teen
protection, in some cases.

“U.S.
</item 22>

<item 23>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
(And on that front, Norway’s DPA, which has a local ban order on Meta’s tracking ads, told us it’s concerned about the subscription plan, saying it has doubts
as to whether it would constitute valid consent.)

As well as needing to comply with the GDPR, which delineates the qualities necessary for consent to be legal (such as that it is specific, informed and freely
given), Meta is now also subject to the pan-EU Digital Services Act (DSA) — which also sets conditions on larger platforms when it comes to tracking and
profiling people for ads. So it won’t just be data protection authorities making a call on whether Meta’s subscription or tracking offer flies; the European
Commission is responsible for oversight of very large online platforms’ DSA compliance.

Meta has also been designated as a so-called gatekeeper, under the DSA’s sister regulation, the Digital Markets Act (DMA) — which also puts some limits on use
of people’s data for ads. The Commission is the sole enforcer of the DMA.
</item 23>

<item 24>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Most recently, Instagram users accused Meta of “shadowbanning” their Stories about Palestine.

It’s the latest in a lengthy history of incidents on Meta platforms that reflect an inherent bias against Palestinian users in its processes, as documented by
years of complaints from both inside and outside the company. The company may not intentionally suppress specific communities, but its moderation practices
often disproportionately affect Palestinian users.

For instance, Meta struggles to navigate the cultural and linguistic nuances of Arabic, a language with over 25 dialects, and has been criticized for neglecting
to adequately diversify its language resources. The company’s black-and-white policies often preclude it from effectively moderating any nuanced topic, like
content that discusses violence without condoning it. Advocacy groups have also raised concerns that Meta’s partnerships with government agencies, such as the
Israeli Cyber Unit, politically influence the platform’s policy decisions.
</item 24>

<item 25>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
A newly unredacted version of the multi-state lawsuit against Meta alleges a troubling pattern of deception and minimization in how the company handles kids
under 13 on its platforms. Internal documents appear to show that the company’s approach to this ostensibly forbidden demographic is far more laissez-faire than
it has publicly claimed.

The lawsuit, filed last month, alleges a wide spread of damaging practices at the company relating to the health and well-being of younger people using it. From
body image to bullying, privacy invasion to engagement maximization, all the purported evils of social media are laid at Meta’s door — perhaps rightly, but it
also gives the appearance of a lack of focus.

In one respect at least, however, the documentation obtained by the attorneys general of 42 states is quite specific, “and it is damning,” as AG Rob Bonta of
California put it. That is in paragraphs 642 through 835, which mostly document violations of the Children’s Online Privacy Protection Act, or COPPA.
</item 25>

<item 26>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta forbids support of terrorist organizations, like most social media companies based in the U.S., but struggles to moderate content around it, from user
discourse to journalistic updates. This policy, along with the company’s partnership with Israel to monitor posts that incite violence, complicates things for
Palestinians living under Hamas’ governance. As EFF points out, something as simple as Hamas’ flag in the background of an image can result in a strike.

Jillian York, the director for international freedom of expression for EFF, blames automation and decisions made by “minimally trained humans” for the
inconsistency. Meta’s zero tolerance policy and imprecise enforcement often suppress content from or about conflict zones, she said. The site’s moderation
issues have negatively affected multiple non-English speaking regions, including Libya, Syria and Ukraine.

“These rules can prevent people from sharing documentation of human rights violations, documentation of war crimes, even just news about what’s happening on the
ground,” York continued.
</item 26>

<item 27>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.
</item 27>

<item 28>
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
</item 28>

<item 29>
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
</item 29>

<item 30>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Following a content moderation warning from European Union regulators earlier this week, Meta has published an overview of how its responding to risks on its
social media platforms stemming from the Israel-Hamas war.

Its blog post covers what it frames as “ongoing efforts”, with some existing policies and tools for users rehashed. But the company confirms it’s made some
changes in light of fast moving events in Israel and Gaza.

These include what it says is a temporary expansion of its Violence and Incitement policy in order to prioritize the safety of Israelis kidnapped by Hamas.

Under this change Meta says it will be removing content that “clearly identifies hostages when we’re made aware of it, even if it’s being done to condemn or
raise awareness of their situation”. “We are allowing content with blurred images of the victims but, in line with standards established by the Geneva
Convention, we will prioritize the safety and privacy of kidnapping victims if we are unsure or unable to make a clear assessment,” it added.
</item 30>

<item 31>
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
</item 31>

<item 32>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
These included findings made by the Court of Justice of the European Union when delivering judgment on July 4, 2023 in a case in which the Court examined the
legal bases on which Meta’s processing of users’ data for behavioural advertising purposes is grounded.

“Acting in consultation with its fellow European supervisory authorities, the DPC has been engaged in a detailed regulatory assessment of the consent-based
model since it was first proposed by Meta in July. That exercise is being led by the DPC, reflecting its position as Lead Supervisory Authority for Facebook and
Instagram in Europe. The exercise has not yet concluded, and no findings have been made to date. It is due to be completed shortly, at which point the DPC will
notify Meta if it considers that the manner in which its new user offerings are to be implemented is compatible with Meta’s obligations under GDPR.”

So Meta’s move to offer users a subscription or tracking hasn’t been signed off by data protection authorities as yet — and it’s clear that further regulatory
interventions could follow.
</item 32>

<item 33>
Meta to offer ad-free subscription in Europe in bid to keep tracking other users (TechCrunch, 2023-10-30T13:52:00+00:00)
Even for a user with just one account (on either Facebook or Instagram) the cost for protecting their privacy from Meta’s tracking and profiling would be almost
€120 per year (for web use) or just over €155 (on mobile).

As we reported earlier this month, Meta is relying on a line in a ruling handed down by the bloc’s top court, the CJEU, earlier this year — where the judges
allowed the possibility — caveated with “if necessary” — of an (another caveat) “appropriate fee” being charged for an equivalent alternative service (i.e. that
lacks tracking and profiling). So the legal fight against Meta’s continued tracking and profiling of users will hinge on what’s necessary and appropriate in
this context.

noyb, the European privacy rights group which has driven much of the strategic litigation against Meta’s tracking and profiling, has already — since 2021 — been
challenging similar ‘pay or okay’ practices by news publishers by filing a series of complaints with data protection authorities.
</item 33>

<item 34>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
In this way, Meta goes through great lengths to avoid meaningfully complying with COPPA, looking for loopholes to excuse its knowledge of users under the age of
13 and maintain their presence on the Platform.

The other complaints in the lengthy lawsuit have softer edges, such as the argument that use of the platforms contributes to poor body image and that Meta has
failed to take appropriate measures. That’s arguably not as actionable. But the COPPA stuff is far more cut and dry.

“We have evidence that parents are sending notes to them about their kids being on their platform, and they’re not getting any action. I mean, what more should
you need? It shouldn’t even have to get to that point,” Bonta said.

“These social media platforms can do anything they want,” he continued. “They can be operated by a different algorithm, they can have plastic surgery filters or
not have them, they can give you alerts in the middle of the night or during school, or not.
</item 34>

<item 35>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
For example, will the survey go to under 13 year olds? Since everyone needs to be at least 13 years old before they create an account, we want to be careful
about sharing findings that come back and point to under 13 year olds being bullied on the platform.”

In 2021, another, studying “child-adult sexual-related content/behavior/interactions” (!) said she was “not includ[ing] younger kids (10-12 yos) in this
research” even though there “are definitely kids this age on IG,” because she was “concerned about risks of disclosure since they aren’t supposed to be on IG at
all.”

Also in 2021, Meta instructed a third-party research company conducting a survey of preteens to remove any information indicating a survey subject was on
Instagram, so the “company won’t be made aware of under 13.”
</item 35>

<item 36>
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
</item 36>

<item 37>
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
</item 37>

<item 38>
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
</item 38>

<item 39>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
And it isn’t because leadership wasn’t aware.

That same year, 2018, CEO Mark Zuckerberg received a report that there were approximately 4 million people under 13 on Instagram in 2015, which amounted to
about a third of all 10-12-year-olds in the U.S., they estimated. Those numbers are obviously dated, but even so they are surprising. Meta has never, to our
knowledge, admitted to having such enormous numbers and proportions of under-13 users on its platforms.

Not externally, at least. Internally, the numbers appear to be well documented. For instance, as the lawsuit alleges:

Meta possesses data from 2020 indicating that, out of 3,989 children surveyed, 31% of child respondents aged 6-9 and 44% of child respondents aged 10 to
12-years-old had used Facebook.
</item 39>

<item 40>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
They choose to do things that maximize the frequency of use of that platform by children, and the duration of that use. They could end all this today if they
wanted, they could easily keep those under 13 from accessing their platform. But they’re not.”

You can read the mostly unredacted complaint here.

(This story has been updated with a comment from Meta.)
</item 40>

<item 41>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
During the last violent outbreak between Hamas and Israel in 2021, a report commissioned by Meta and conducted by a third party concluded that the company’s
actions had an “adverse human rights impact” on Palestinian users’ right to freedom of expression and political participation.

Meta communications director Andy Stone declined to comment on the record, and pointed TechCrunch to Meta’s newsroom post about the conflict, which was updated
on Oct. 18 with a statement addressing the shadowbanning claims.

“Our policies are designed to keep people safe on our apps while giving everyone a voice,” the post stated. “We apply these policies equally around the world
and there is no truth to the suggestion that we are deliberately suppressing voice … We can make errors and that is why there is an appeals process for people
to tell us when they think we have made the wrong decision, so we can look into it.”

The belief that Meta shadowbans, or limits the visibility of, content about Palestine is not new.
</item 41>

<item 42>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
One stated goal, according to the lawsuit, was specifically to “grow [Monthly Active People], [Daily Active People] and time spent among U13 kids.”

It’s important to note here that while Meta does not permit accounts to be run by people under 13, there are plenty of ways it can lawfully and safely engage
with that demographic. Some kids just want to watch videos from SpongeBob Official, and that’s fine. However, Meta must verify parental consent and the ways it
can collect and use their data is limited.

But the redactions suggest these under-13 users are not of the lawfully and safely engaged type. Reports of underage accounts are reported to be automatically
ignored, and Meta “continues collecting the child’s personal information if there are no photos associated with the account.” Of 402,000 reports of accounts
owned by users under 13 in 2021, fewer than 164,000 were disabled.
</item 42>

<item 43>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“This was also a global issue that was fixed within a few hours. We understand people rely on these tools and we’re sorry to anyone who felt the impact of these
issues,” the post stated.

It’s unclear if this “bug” impacted accounts posting content unrelated to Gaza. Instagram users have posted screenshots showing that Stories about Palestine
have received significantly fewer views than other Stories posted on the same day, and allege that their view counts went back up when they posted content
unrelated to the conflict.

ofc IG’s hiding my stories about Palestine, y’all see the difference? lmaoaoaoao pic.twitter.com/1iYIIKBhtN — ༒︎ (@arxbprince) October 15, 2023

A user based in Egypt, who asked to stay anonymous for fear of harassment, said her posts usually get around 300 views, but when she started posting
pro-Palestine content after the Hamas attack earlier this month, her stories would only get one to two views.
</item 43>

<item 44>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
In screen recordings posted online, users found that if they included “Palestinian” and the corresponding flag emoji in their Instagram bio along with the
Arabic phrase, Instagram automatically translated their bio to “Palestinian terrorists – Praise be to Allah” or “Praise be to God, Palestinian terrorists are
fighting for their freedom.” When users removed “Palestinian” and the flag emoji, Instagram translated the Arabic phrase to “Thank God.” Instagram users
complained that the offensive mistranslation was active for hours before Meta appeared to correct it.

Shayaan Khan, a TikTok creator who posted a viral video about the mistranslation, told TechCrunch that Meta’s lack of cultural competence isn’t just offensive,
it’s dangerous. He said that the “glitch” can fuel Islamophobic and racist rhetoric, which has already been exacerbated by the war in Gaza. Khan pointed to the
fatal stabbing of Wadea Al-Fayoume, a Palestinian-American child whose death is being investigated as a hate crime.
</item 44>

<item 45>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
When reached for comment about these recent claims of shadowbanning, a representative for the company pointed TechCrunch to a Threads post by Stone.

“We identified a bug impacting all Stories that re-shared Reels and Feed posts, meaning they weren’t showing up properly in people’s Stories tray, leading to
significantly reduced reach,” Stone said. “This bug affected accounts equally around the globe and had nothing to do with the subject matter of the content —
and we fixed it as quickly as possible.”

But many are frustrated that Meta continues to disproportionately suppress Palestinian voices. Leen Al Saadi, a Palestinian journalist currently based in Jordan
and host of the podcast “Preserving Palestine,” said she is used to “constantly being censored.” Her Instagram account was restricted last year after she posted
a trailer for the podcast’s first episode, which discussed a documentary about Palestinian street art under occupation.

“Palestinians are currently undergoing two wars,” Al Saadi said. “The first is with their legal occupier.
</item 45>

<item 46>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta did not respond to TechCrunch’s request for comment about the mistranslation. Abukhater said that Meta told 7amleh that a “bug” caused the mistranslation.
In a statement to 404 Media, a Meta spokesperson said that the issue had been fixed.

“We fixed a problem that briefly caused inappropriate Arabic translations in some of our products,” the statement said, “We sincerely apologize that this
happened.”

As the war continues, social media users have tried to find ways around the alleged shadowbanning on Instagram. Supposed loopholes include misspelling certain
words, like “p@lestine” instead of “Palestine,” in hopes of bypassing any content filters. Users also share information about Gaza in text superimposed over
unrelated images, like a cat photo, so it won’t be flagged as graphic or violent content.
</item 46>

<item 47>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
His argument essentially was that the improvements and safety features Meta was implementing were not enough and not effective.

However, a policy position like the one Meta announced today is not something that would have been rushed out in a matter of days as a PR response to Bejar’s
claims, but something Meta has been stewing on for some time. After all, the app stores already have age guidelines, so it would seem to follow that they should
also enforce them, if enforcement is to be regulated — or that seems to be Meta’s thinking.

If anything, the call to bring the app stores into the fight is a response to the lack of action in Congress to pass online child safety regulations, like the
hotly debated KOSA (the Kids Online Safety Act) leaving U.S. states to individually create laws due to a void of federal guidelines.

Utah, for instance, passed a measure that will require social networks to give Utah parents access to their children’s posts, messages and comments, and would
block social media access during certain hours.
</item 47>

<item 48>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
“U.S. states are passing a patchwork of different laws, many of which require teens (of varying ages) to get their parent’s approval to use certain apps, and
for everyone to verify their age to access them,” she wrote. “Teens move interchangeably between many websites and apps, and social media laws that hold
different platforms to different standards in different states will mean teens are inconsistently protected,” Davis said.

Meta’s proposed “better way” of requiring parents to manage a teen’s app downloads pushes the burden back to the app stores and to parents, who often today do
not monitor or manage their teens’ smartphone use. But while it’s true that the app stores and smartphone makers would know the ages of their users, requiring
parents to approve apps is the equivalent of overriding parental discretion and choice.

Other entertainment ratings — like those for TV, movies or video games — are meant to be guidelines for parents, not laws. Meta’s position that apps are somehow
different, then, is a fairly extreme position to take.
</item 48>

<item 49>
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
</item 49>

<item 50>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.
</item 50>
=== END ITEM id=d5cdcdbefc952e9e ===

=== ITEM id=919b4f6d8752412c ===
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
– UCLA misses the field goal! RJ Lopez's 24-yard attempt hits the left upright and is no good. The Bruins keep a one-point lead with a 1:37 left in the second
quarter.

9:03 p.m. PENALTY – Shilo Sanders is ejected for targeting after a hit on Carsen Ryan. He will be out for the remainder of the game.

Shilo Sanders was disqualified for targeting after this hit pic.twitter.com/1EJ7IOdKKs — FOX College Football (@CFBONFOX) October 29, 2023

8:58 p.m. – Colorado goes three-and-out, unable to score on any turnovers. For UCLA, Garbers is back, and he takes a 17-yard run to the UCLA 37 for a first
down.

8:51 p.m. INTERCEPTION – Third times a charm! Collin Schlee, who came in for Garbers, is interception by Hunter.
</item 1>

<item 2>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado fans, you're back. For now.

A 43-yard pass from Shedeur Sanders on the Buffaloes' final drive helped set up Alejandro Mata for the game-winning 43-yard field goal with 12 seconds left,
delivering Colorado the 27-24 win over Arizona State for their first Pac-12 win of the season. It snaps their eight-game conference losing streak.

Sanders had a mediocre day by his lofty standards, but his 239 yards and a touchdown were enough in perhaps Colorado's most crucial game so far. It brought the
team to 4-2, but had it lost, it would have been much harder for the Buffs to finish the season bowl eligible. A win gives them a leg up in a competitive
conference, particularly facing four ranked teams down the stretch of the season.

If Colorado wants to solidify a bowl game, it'll have to fix many things, one of them being the offensive line.
</item 2>

<item 3>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
The QB keeps it for 16 yards and a touchdown. Less than a minute is left until halftime.

8:09 p.m. – Sanders finds Jimmy Horn Jr. for 24 yards. Colorado is within striking distance, the ASU 39, with only two minutes left.

8:06 p.m. – ASU's head coach Kenny Dillingham could not be more unhappy with the referees, who ruled Bourguet's run just short of a first down. It's fourth down
for the Sun Devils.

Colorado 7, Arizona State 14

7:48 p.m. TOUCHDOWN – Cam Skattebo pushes through the huddle for the score! It's his fifth rushing touchdown of the season. ASU now leads 14-7 with 6:38 left in
the half.

7:45 p.m. – Andre Johnson reaches over the pylon with one hand, but the ball is knocked out. The play is being reviewed for a touchdown.
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Three of the seven teams passed, resulting in one first down, one sack and Hurts’ interception. Of these three teams, only the Eagles lost the game.

3. The 49ers lost Sunday when their rookie kicker missed from 41 yards. Is San Francisco wise trusting a championship roster to a rookie’s foot come playoff
time?

The 49ers raised eyebrows among some when they used their second-highest pick in the 2023 draft — a third-round choice, 99th overall — for a kicker. The
selection of Michigan’s Jake Moody gave the 49ers a cost-effective successor to veteran Robbie Gould. General manager John Lynch called Moody a potentially
“foundational” player whose stock in the draft was so high, other teams were trying to trade ahead of San Francisco for a shot at him.

Who do you want lining up the potential winning kick in a Super Bowl?
</item 4>

<item 5>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
You just might not have a better option than Fields.

LAST WEEK: Posted his first-ever game with four passing touchdowns and over 300 yards against a Broncos defense that was devoid of a serious pass rush or pass
coverage skills through the first three quarters. By the fourth quarter the Broncos were generating heavy pressure and Fields finished 5 of 11 for 50 yards and
no touchdowns with two turnovers.

FILM: For three quarters Fields looked terrific firing downfield from clean pockets with plenty of great reads and accurate throws along with a play or two
where he manipulated defenders with the threat of him running only to throw over their heads, including on Kmet's short touchdown. His off-target rate was
literally 0.0%. But by the fourth only some plays were fine -- on others he either was pressured, threw off his back foot, threw with too much velocity or
wasn't on the same page as his target.
</item 5>

<item 6>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Its 7-0 just a few minutes in.

6:35 p.m. – The game will begin on the Pac-12 Network Arizona due to the Washington State vs. UCLA game running long.

6:10 p.m. – Starting safety Trevor Wood is out for Colorado. Good news is Shilo Sanders will start.

6:05 p.m. – The Sun Devils are looking to end their eight-gam losing streak against FBS teams. Ironically, their last FBS win was against Colorado when Trenton
Bourguet set a school record for most passing yards in a career start. He's expected to be Arizona State's starting QB tonight.

5:50 p.m. – Colorado to debut new helmets in Tempe.

5:20 p.m. – The Buffs have entered the chat.

MORE: Best Colorado player prop odds for Week 6 college football matchup against Arizona State

How to watch Colorado vs. Arizona State

Date : Saturday, Oct.
</item 6>

<item 7>
Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game (Sporting News, 2023-12-24T23:00:03+00:00)
The Christmas Eve showdown between the Patriots and the Broncos was expected to be a solid victory by Denver against a team that had won just three games
heading into the contest. It was anything but that.

In a game that it had it all between critical turnovers and scores that came down to the final minute, Patriots kicker Chad Ryland hit a 56-yard field goal to
send the Patriots to a 26-23 victory.

Coming into this game, New England had played the Chiefs tough but lost at home to fall to 3-11 and improve its chances of landing the No. 1 pick in next year's
draft. Denver aimed to get back to the right side of the win column after getting blown out by the Lions in an all-around abysmal performance.

MORE: Chad Ryland redeems himself with game-winning field goal

In this one, it took the Broncos a while once more to put points on the board, with 16 of them coming in the fourth quarter in a comeback effort that proved
just a hit too late.
</item 7>

<item 8>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
10:01 p.m.: The Vikings are moving the chains. They've picked up two first downs in five plays, gaining 43 yards in the process. Alexander Mattison picked up 23
yards on the ground, and Joshua Dobbs and Jordan Addison connected on an 18-yard pass to pick up the bulk of the yardage.

9:56 p.m.: A holding penalty derailed the Broncos' drive as they came up way short on a 3rd & 20 attempt. Minnesota will take over from its own 31 after a
19-yard punt return from Brandon Powell.

9:54 p.m.: Denver receives to start the second half and we're back in action.

End of first half: Vikings 10, Broncos 9

9:38 p.m.: FIELD GOAL — Will Lutz from 52 yards... it's good! The Broncos kicker steps up to bury a long field goal as the first half comes to a close.
</item 8>

<item 9>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Vikings decided to go for it and Joshua Dobbs picked up the first down on a QB sneak.

10:54 p.m.: Ty Chandler continues to rumble. The spell RB picked up a massive gain on 3rd & 20 and it appeared that he got the first down, but the spot on the
field is under review. This could be a crucial call.

10:48 p.m.: It looked like it was going to be a three-and-out for the Vikings offense but a fake punt has kept the drive alive! Ty Chandler picked up 31 yards
on a gutsy call from head coach Kevin O'Connell to bring the ball into Denver's territory.

Vikings 17, Broncos 15

10:43 p.m.: FIELD GOAL — Will Lutz knocks one in from 28 yards, his fifth made field goal of the day. Lutz has been the Broncos' lone source of offense,
bringing Denver within one score with 10:37 to play in the game.
</item 9>

<item 10>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
GUTSY.



After two QB scrambles, Ethan Garbers finds Carson Steele for the wide-open TD 🙌



UCLA 7, Colorado 6

14:19 – 2nd quarter



📺: ABC

📲: https://t.co/rTM1Y5TWC9#GoBruins pic.twitter.com/4ZzaQVuu1P — UCLA Football (@UCLAFootball) October 29, 2023

End of first quarter: Colorado 6, UCLA 0

8:22 p.m. – Garbers is sacked! Jordan Domineck takes down the Bruin to force 4th &11. But UCLA goes for it, and Garbers finds the first down himself on a
13-yard run.

Colorado 6, UCLA 0

8:05 p.m. – Colorado settles for a field goal again, this time for a 39-yarder.
</item 10>

<item 11>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Can the Buffaloes finally connect for a touchdown before its too late?

UCLA adds on another 💪@UCLAFootball makes it 21-9 in the 4th quarter vs Colorado pic.twitter.com/6Eu9gn6kgn — FOX College Football (@CFBONFOX) October 29, 2023

End of third quarter: Colorado 9, UCLA 14

10:20 p.m. – What a catch for the Bruin! After review, Garbers pass to Loya is ruled complete.

Colorado 9, UCLA 14

10:06 p.m. FIELD GOAL – Mata nails a 34-yard field goal to lessen Colorado's deficit to five. Four minutes remain in the third quarter.

10:03 p.m. – Sacked! Sanders is taken down by Laiatu Latu, who has multiple tonight. UCLA records its sixth sack of the night with the latest.
</item 11>

<item 12>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
UCLA records its sixth sack of the night with the latest.

Colorado 6, UCLA 14

9:40 p.m. TOUCHDOWN – Bruins bouncing back quick after halftime! Garbers finds Logan Loya and Moliki Matavao on two plays, and UCLA has another touchdown on the
board.

Halftime: Colorado 6, UCLA 7

9:15 p.m. FUMBLE – Turnover No. 4 for the Bruins. Steele fumbles on his run again, and it's recovered by Cam'Ron Silmon-Craig in Colorado territory. While the
Buffaloes have prevented more points UCLA, they've only scored three off turnovers. Twenty seconds remain in the half.

The ball pops out and @CUBuffsFootball has it! 🦬🔒 pic.twitter.com/tV7FH3moiu — FOX College Football (@CFBONFOX) October 29, 2023

9:07 p.m. – UCLA misses the field goal!
</item 12>

<item 13>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Colorado vs. UCLA score

Q1 Q2 Q3 Q4 F Colorado 6 0 3 7 16 UCLA 0 7 7 14 28

Colorado vs. UCLA results, highlights from Week 9 game

Final: Colorado 16, UCLA 28

Colorado 16, UCLA 28

10:58 p.m. TOUCHDOWN – The Buffaloes aren't going down without a fight. Sanders gets his first touchdown of the night, finding Jimmy Horn Jr. in the end zone on
an 18-yard toss. Colorado attempts an onside kick with only two-and-a-half minutes left, but the Bruins recover it.

Colorado 9, UCLA 28

10:45 p.m. TOUCHDOWN – Collin Schlee finds the end zone! His seven-yard run helps add seven points for the Bruins, who lead 28-9 with six minutes left in the
game.
</item 13>

<item 14>
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
</item 14>

<item 15>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
It didn't help that the Colorado running game was absent in the second half. Sanders handed the ball off only four times through the final two quarters, with
the first one coming halfway into the final frame; the Buffaloes scored their only touchdown on that drive. They finished with 38 rushing yards, the lowest in a
game this season.

Safety Shilo Sanders missed much of the game after being ejected late in the first half for targeting, but Travis Hunter shined in his second game back on the
field after injury. He recorded the game's two picks.

Colorado's quest for bowl eligibility continues after dropping to 4-4. Two of the Buffs' final four opponents -- Oregon State and Utah, are ranked.

MORE: Watch Colorado vs. UCLA live with Fubo (free trial)

The Sporting News tracked highlights and updates from the Colorado-UCLA game.

Colorado vs.
</item 15>

<item 16>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
But O'Connell way overthrows Adams, and the Raiders have to settle for the 40-yard field goal to tie the game with 2:36 left in the third quarter. Adams was
furious there wasn't a penalty. All kicks so far tonight. Jets 9, Raiders 9

10:27 p.m.: O'Connell rifles a quick pass out to Hunter Renfrow, who races up and gets near the line of scrimmage but is just inches shy of the line. The
Raiders opt to go for it and hand the ball to Jacobs, who finds a hole on the right side and gets the necessary yardage to the Jets 23 for the first.

10:26 p.m.
</item 16>

<item 17>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Arizona State

Date : Saturday, Oct. 7

: Saturday, Oct. 7 Time : 6:30 p.m. ET

: 6:30 p.m. ET Location : Mountain America Stadium in Tempe, Arizona

: Mountain America Stadium in Tempe, Arizona TV channel : Pac-12 Network

: Pac-12 Network Live stream: Pac-12 Network, Fubo

It'll be a sold out Mountain America Stadium in Tempe. Colorado-Arizona State kicks off at 6:30 p.m. ET on Saturday, Oct. 7. Viewing is available only on Pac-12
Network for those who have the cable extension. Streaming the game is available through Fubo, which offers a free trial.
</item 17>

<item 18>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Who do you want lining up the potential winning kick in a Super Bowl? It’s an interesting question in a season that has seen Moody and Saints rookie Blake Grupe
miss game-winning tries, but the data could be on the 49ers’ side.

FGA Dist Rookies Veterans <33 Yds 97% [31-32] 94% [330-351] 34-44 Yds 75% [21-28] 82% [230-279] 45-50 Yds 78% [7-9] 73% [121-165] 51-55 Yds 100% [8-8] 59%
[33-56] 56+ Yds 0% [0-1] 55% [6-11] Totals 86% [67-78] 84% [720-862]

The table above shows field-goal make rates in the postseason since 2000 for rookie kickers vs. veterans.
</item 18>

<item 19>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado takes over on the ASU 25.

SHEDEUR TO ANTONIO FOR A HUGE GAIN 🦬@CUBuffsFootball pic.twitter.com/tvfWjRWWAE — FOX College Football (@CFBONFOX) October 8, 2023

Colorado 24, Arizona State 24

10:11 p.m. TOUCHDOWN – Don't count the Sun Devils out yet! Bourguet finds Troy Omeire on a 15-yard pass to tie it up. Only 50 seconds remain.

9:48 p.m. – It's blocked! Carter Stoutmire gets in front of the 52-yard field goal attempt from Longhetto to deny a Sun Devil score.

9:38 p.m. FIELD GOAL – Alejandro Mata nails a 42-yard field goal to extend Colorado's lead to seven. Ten minutes are left in the game.

9:35 p.m. – That's a huge run for the Buffaloes!
</item 19>

<item 20>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
veterans. The numbers stack up pretty evenly across various distance cutoffs. Based on these numbers, there’s no statistical indication rookies are less
effective in the postseason.

“Playoff teams have a different aura, different feel in the locker room,” a veteran evaluator said. “Percentages are one thing, but who is the guy who trots out
there to try the game-winner? It’s a different pressure shooting free throws up by 20, down by 20, than in that type of situation.”

Advertisement

Fans might recall rookie kickers missing playoff kicks: Rodrigo Blankenship with the Colts in 2020, Harrison Butker with the Chiefs in 2017, Nate Kaeding with
the Chargers in 2005. But the NFL has seen plenty of rookies kick effectively in the postseason as well. Evan McPherson made all 14 postseason tries with the
2021 Super Bowl Bengals. The 2017 Eagles won a Super Bowl with rookie Jake Elliott performing well.
</item 20>

<item 21>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
– Colorado settles for a field goal again, this time for a 39-yarder. It leads UCLA 6-0 with 7:44 remaining in the first quarter.

7:59 p.m. INTERCEPTION – Welcome to Hollywood, Travis Hunter! The two-way star picks off Ethan Garbers at the Bruins' 36-yard line. After review, the
interception stands, and it's Buffalo ball. Colorado defense with a statement play early on.

WHAT A PICK BY TRAVIS HUNTER 🔥@CUBuffsFootball pic.twitter.com/3i9twSG1uq — FOX College Football (@CFBONFOX) October 29, 2023

Colorado 3, UCLA 0

7:54 p.m. FIELD GOAL – Sanders can't find the end zone, but Alejandro Mata finds space between the uprights. His 31-yard field goal is good, and the Buffs take
the first lead of the game on their first drive.

7:52 p.m.
</item 21>

<item 22>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
PENALTY – Sun Devils Ed Woods is ejected for targeting. He will miss the remainder of this game as well as the first half against Washington in two weeks.

9:10 p.m. – That's the fourth sack allowed by the Colorado O-line. The Buffaloes now lead FBS with most sacks allowed this season at 30.

8:52 p.m. – Skattebo is heating up under the sun. He's added 24 rushing yards from Arizona State's possession.

End of second quarter: Colorado 14, Arizona Sate 17

Colorado 14, Arizona State 17

8:20 p.m. FIELD GOAL – Dario Longhetto nails a 33-yard field goal to end the half. The Sun Devils have a narrow lead over the Buffs at the halfway mark.

Colorado 14, Arizona State 14

8:14 p.m. TOUCHDOWN – Sanders ties it up! The QB keeps it for 16 yards and a touchdown.
</item 22>

<item 23>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
COURTLAND SUTTON INCREDIBLE 4TH DOWN GRAB ‼️



🎥: @NFL

pic.twitter.com/T0Mrch9Tf9 — The Sporting News (@sportingnews) November 20, 2023

Vikings 20, Broncos 15

11:02 p.m.: FIELD GOAL — Greg Joseph knocks it in from 30 yards. The Vikings' offense was moving, and it did its job by chewing a lot of clock on a 14-play,
seven-minute and 20-second drive. Minnesota stalled out on Denver's 12-yard line, settling for a field goal.

Can the Vikings' defense keep the Broncos out of the end zone as it has all night? The drive of the game is coming up.

10:57 p.m.: The spot on the field was overturned! Ty Chandler was ruled just short of the line to gain, setting up a 4th & inches.
</item 23>

<item 24>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
10:42 p.m.: A 15-yard facemask penalty on Broncos WR Marvin Mims Jr. backed up the offense and the Vikings' defense held strong again as a result. Here comes
Will Lutz for his fifth field goal attempt of the game.

10:38 p.m.: INTERCEPTION! Joshua Dobbs was hit as he threw and Broncos CB Ja'Quan McMillan came up with the pick. It was LB Josey Jewell who brought the
pressure on Dobbs. Denver takes over from the Vikings' 9-yard line after McMillan picked up nine yards on the interception return.

Vikings 17, Broncos 12

10:34 p.m.: FIELD GOAL — Will Lutz buries his fourth field goal of the day, this time from 37 yards. Lutz has been automatic in this one, going a perfect
4-for-4 to account for all of Denver's points.

10:32 p.m.
</item 24>

<item 25>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
9:35 p.m. – That's a huge run for the Buffaloes! Weaver returns it 51 yards, only stopping after tripping himself up. Colorado is 25 yards away from another
touchdown.

Colorado 21, Arizona State 17

9:26 p.m. TOUCHDOWN – Colorado pulls ahead! Sanders finds Javon Antonio for a 9-yard touchdown, his first of the year. It's now a four-point lead for Colorado
with 14 minutes to go.

Buffs are up thanks to @7asavage



CU 21, ASU 17



📺 Pac-12 Network pic.twitter.com/3R8w2Qdm9e — Colorado Buffaloes Football (@CUBuffsFootball) October 8, 2023

End of third quarter: Colorado 14, Arizona State 17

9:13 p.m. PENALTY – Sun Devils Ed Woods is ejected for targeting.
</item 25>

<item 26>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
UCLA topped Colorado 28-16 at the Rose Bowl in Deion Sanders and Chip Kelly's only Pac-12 matchup.

The Bruins improved to 6-2, clinching bowl eligibility and keeping them alive in the Pac-12 race with a 3-2 league mark.

In his third start of the season, Bruins' QB Ethan Garbers passed for 269 yards and two touchdowns. He and Collin Schlee combined for two interceptions in
UCLA's first-half turnover-fest. It had four, including two fumbles by Carson Steele. Luckily for Kelly, his team fixed the issue and did not have any turnovers
in the second-half.

Colorado, on the other hand, could not figure out how to protect QB Shedeur Sanders, who tossed for 217 yards and one touchdown. He was sacked seven times,
tying a season-high.

It didn't help that the Colorado running game was absent in the second half.
</item 26>

<item 27>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
It allowed five sacks on Sanders, bringing the season total to a nation-leading 31. The defense picked up five sacks themselves to maintain pressure on Trenton
Bourguet.

For Arizona State, Bourguet put all he could into Saturday's narrow loss. He's one of two healthy quarterbacks on the roster, and passed for 335 yards and a
touchdown. The Sun Devils had a real shot at their first conference win. But as the game came to a close, Bourguet seemed roughed up and slow, though he did
drive the team down the field 94 yards in 13 plays for the game-tying touchdown with 50 seconds left.

Luckily for Arizona State, next week is a bye week. What's not so lucky is that it'll go on to play three ranked teams over the following three weeks.

We didn't see the worst of either team, though we didn't see the best either. Colorado needed a bounce back win and got one.
</item 27>

<item 28>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
The play is being reviewed for a touchdown.

7:41 p.m. – Colorado is called offsides, giving Arizona State a first down at the Buffs' 48.

7:34 p.m. – After scoring a touchdown on its opening possession, Arizona State has three straight three-and-outs.

7:30 p.m. – The Sun Devils record their third sack of the day already, this time forcing fourth down for the Buffaloes. Safe to say their defense came prepared.

End of first quarter: Colorado 7, Arizona State 7

Colorado 7, Arizona State 7

7:17 p.m. TOUCHDOWN – The Buffs tie it up! Needing one yard on the fourth down, Sanders hands it off to Xavier Weaver who weaves his way into the end zone. Its
7-7 in Tempe with two minutes remaining in the first quarter.

On 4th & goal Xavier Weaver takes it in for @CUBuffsFootball 🦬
</item 28>

<item 29>
Chiefs vs. Jets live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-01T23:47:00+00:00)
Jets LB CJ Mosley picks off Patrick Mahomes on the first play of the drive. He was looking at TE Travis Kelce on a 35-yard pass but Mosley read it perfectly and
Mahomes underthrew his target. The Jets take over from their own 35 with two timeouts and 18 seconds on the clock.

9:46 p.m.: MISSED FIELD GOAL. Greg Zuerlein's 52-yard attempt hits off the upright to bring a disappointing end to the Jets' drive. Patrick Mahomes and the
Chiefs will have 26 seconds and two timeouts from their own 34 to try and make something happen before the end of the half.

9:43 p.m.: Zach Wilson is marching the troops! The Jets QB is putting together an impressive two-minute drill. He has led an eight-play, 41-yard drive to bring
the Jets to Kansas City's 34 yard line.

Chiefs 20, Jets 12

9:36 p.m.: FIELD GOAL.
</item 29>

<item 30>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
On 4th & goal Xavier Weaver takes it in for @CUBuffsFootball 🦬



This game is tied 💪 pic.twitter.com/bGwGZ7b0IF — FOX College Football (@CFBONFOX) October 7, 2023

7:06 p.m. – Anthony Hankerson gets two first downs for the Buffs on a four-yard run and a 12-yard run.

7:00 p.m. – It was a three-and-out for Colorado, and now it's a three-and-out for Arizona State.

6:54 p.m. – Intentional grounding is called on Shedeur Sanders on first down.

Colorado 0, Arizona State 7

6:47 p.m. TOUCHDOWN – The Sun Devils start quick, scoring a touchdown on its first drive behind Elijhah Badger's 48 yards. Its 7-0 just a few minutes in.

6:35 p.m.
</item 30>
=== END ITEM id=919b4f6d8752412c ===

=== ITEM id=60b34adffe29d3a9 ===
Gold sentences and facts:
Sentence 0: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 0.0: The trial's standout testimony in the second week came from Caroline Ellison.
  Fact 0.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 0.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 0.3: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 0.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 0.5: Caroline Ellison claimed she did this under the instruction of SBF.
Sentence 1: Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).
  Fact 1.0: Bankman-Fried was even compared to Warren Buffet.
  Fact 1.1: Many people called Bankman-Fried the white horse of crypto.
  Fact 1.2: TechCrunch never called Bankman-Fried the white horse of crypto.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth.
  Fact 2.2: The prosecution said Bankman-Fried committed fraud to achieve great power.
  Fact 2.3: The prosecution said Bankman-Fried committed fraud to achieve great influence.
  Fact 2.4: The defense countered that the FTX founder acted in good faith.
  Fact 2.5: The defense said the FTX founder never meant to commit fraud.
  Fact 2.6: The defense said the FTX founder never meant to steal.
  Fact 2.7: The defense said the FTX founder basically got in over his head.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 2>

<item 3>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 7>

<item 8>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 10>

<item 11>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 11>

<item 12>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 12>

<item 13>
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
</item 13>

<item 14>
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
</item 14>

<item 15>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 15>

<item 16>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 16>

<item 17>
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
</item 17>

<item 18>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
The trial of Sam Bankman-Fried is likely to be more consequential than just whether the man himself is found guilty. Depending on what evidence is introduced
during the trial, it could be rough for the entire crypto industry.

“How much damage can this trial do to the already beaten-down reputation of the industry at this point?” asks Yesha Yadav, a law professor at Vanderbilt
University. “This trial is going to be an excruciating moment for the industry because no one knows what kind of evidence might come out.”

Bankman-Fried, the founder of FTX and Alameda Research, is facing seven counts of criminal charges: two counts of wire fraud, and five counts of conspiracy
charges. FTX was a failed cryptocurrency exchange founded in 2019. According to a now-deleted profile from FTX investors Sequoia Capital, FTX was founded
because of Bankman-Fried’s frustration with other exchanges when he was running Alameda Research, his crypto trading firm.
</item 18>

<item 19>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 19>

<item 20>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 20>

<item 21>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 21>

<item 22>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 22>

<item 23>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 23>

<item 24>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 24>

<item 25>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
First and foremost, they’re trying to prove their client is not guilty. But just in case they don’t get the outcome they want, they’re also laying the
groundwork for appeals and sentencing arguments. Any piece of evidence they want to introduce for those two purposes has to come into play in the trial.

Already, Bankman-Fried’s lawyers are marking arguments that his constitutional rights are being violated by his pretrial incarceration because, among other
things, his internet connection wasn’t good enough for him to plan his defense, notes LaVigne. That could be grounds for an appeal.

“Is he going to throw the entire industry under the bus?”

Bankman-Fried’s behavior after the fall of FTX suggests he’s something of a wild card. He may suggest he was acting on the advice of his lawyers. But he may
also introduce other evidence that could be troublesome — implying, for instance, that he was engaged in standard industry behavior or that everything that
happened was Binance’s fault.
</item 25>

<item 26>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
He could face up to 115 years in jail if convicted on all charges.

The crypto industry as a whole suffered from FTX’s collapse, which was the first of many. BlockFi filed for Chapter 11 in November 2022, as did Genesis Global
Trading in January.

Where we are today

Bankman-Fried will be represented by Cohen & Gresser, and Mark Cohen, a high-profile defense attorney and former federal prosecutor, will be the lead attorney.
If that name sounds familiar, it might be because he also represented Ghislaine Maxwell in her sex trafficking trial related to Jeffrey Epstein. He requested an
early release for Bankman-Fried but was denied.

With the trial beginning, we’re witnessing how FTX’s story ends. But what’s on our mind is what happens to the investors and creditors affected by the collapse?
And what happens to the billions in crypto assets tied up in legal proceedings?
</item 26>

<item 27>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 27>

<item 28>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 28>

<item 29>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 29>

<item 30>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.
</item 30>
=== END ITEM id=60b34adffe29d3a9 ===
