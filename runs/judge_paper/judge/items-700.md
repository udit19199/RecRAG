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

=== ITEM id=271739166f813f64 ===
Gold sentences and facts:
Sentence 0: But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
  Fact 0.0: The speaker thinks that people will look back and see that the past year is a really important turning point for the role social media plays in people's lives.
  Fact 0.1: The speaker adds this.
Sentence 1: The Goellers are now seeking support again, launching the "Save Lucas" campaign across social media.
  Fact 1.0: The Goellers are now seeking support again.
  Fact 1.1: The Goellers are launching the 'Save Lucas' campaign.
  Fact 1.2: The 'Save Lucas' campaign runs across social media.

Retrieved items:
<item 1>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Despite the problems with social media and the urgency, in terms of its impact on society, Ohanian believes there’s still some hope. He noted, for example, that
the founders his fund 776 invests in are much smarter and more aware of the issues than he ever was at their age. Perhaps the next platforms will still be
addictive, but in a way that doesn’t “further disrupt democracy,” he mused.

He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not. “I think we’ll
see a rebirth of live gatherings and in-person experiences…the bright side of this is that I hope it then gets us back in touch with our humanity, and the
things that no amount of AI can replace.”

“I want to be an optimist about it,” Ohanian added. “And I think there’s still a path forward but we’re fighting against very, very strong human things.
</item 1>

<item 2>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
Brands, too: Remember the Tony the Tiger VTuber in August of 2022?

Many creators that jumped on the trend have since stopped using their avatars, but at the start of the COVID-19 pandemic, there was a large increase in VTubers
on Twitch and other streaming platforms, like YouTube, as models and equipment became more accessible to the general public. Twitch has since gone on to
highlight a wide variety of VTubers through its VTuber Takeover promotion, which most recently ran in August. Twitch also reported that VTubing-related content
on the platform shot up by some 350% between January and August 2022. Despite this, some have argued that the VTubing bubble has all but burst.

Polygon is celebrating TwitchCon 2023 with a package of stories covering streaming trends and news from the show. Check it out!

“I do feel that VTubing has become less novel as the market has become more saturated,” says a streamer with agency Nijisanji who is not approved to speak
publicly.
</item 2>

<item 3>
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
</item 3>

<item 4>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
To that end, Pebble’s founding team included co-founder Sarah Oh, Twitter’s former human rights advisor.

Pebble continues to believe its approach to moderation was correct, even if it was not ultimately a factor in driving growth.

“We came in with a particular angle: kinder, safer. Trust and safety. And I think our approach to moderation that we did have on the site did work,” says
Cselle, adding that Pebble didn’t have some of the issues other federated platforms faced. For example, TechCrunch reported that Bluesky earlier this year had
strained its relationship with Black users, as it failed to crack down on bigotry on the site. Later, users on Bluesky were even creating usernames with racial
slurs, and yet the platform didn’t issue a public apology. Pebble never faced any such controversies.

But it may have positioned itself a little too far into the “kindness” territory, the founders now believe.
</item 4>

<item 5>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
It also gave Flipboard a way to stay connected to social media after Twitter/X increased its API fees for third-party developers, making it unsustainable for
many developers to continue to work with the company.

The Flipboard app’s primary purpose has been to curate into social “magazines” news and information found on the web, including links to articles, photos and
other social posts. As a result, it had relied on Twitter as one of its sources of information. That changed this year, when Flipboard shifted its Twitter
integration over to Mastodon and another alternative social app, Bluesky. It also set up its own Mastodon server and began curating news across the fediverse
via editorial “desks” focused on improving the discovery of news on Mastodon.

All this was in the lead-up to making Flipboard itself a federated social app, a process that’s kicking off today.
</item 5>

<item 6>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Superpedestrian

Is shutting down its U.S.-based shared scooter operation and exploring a sale of its European business, TechCrunch has exclusively learned.

Bolt

Confirmed it laid off 29% of its staff December 15. The e-commerce company known for its one-click checkout technology was at one time the subject of a federal
probe.

Cruise

In an announcement obtained exclusively by TechCrunch, the embattled self-driving car company is cutting 900 employees, or 24% of its workforce, per a December
14 email from new president and CTO Mo Elshenawy.

Etsy

Is laying off 11% of its workforce. In a letter to employees December 13, CEO Josh Silverman cited a “very challenging macro and competitive environment.”

Chipper Cash

Has reportedly laid off 15 people across various departments. It would be the fintech unicorn’s fourth round of layoffs this year.

SmileDirectClub

Has shut down three months after filing for Chapter 11 bankruptcy, citing $900 million of debt.
</item 6>

<item 7>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
With Amo, the founder of Zenly wants to make social apps social again The startup raised $20 million and is launching ID today

In 2018, I wrote a TechCrunch article stating that 2018 was “the year social networks were no longer social.” Reflecting on that article, I’m not sure that 2018
was the turning point. But the premise of the article still holds up well.

At some point, social networks were no longer about connecting with your closest friends, keeping up with long-distance family members and feeling a special
connection with people you love.

TikTok, Instagram, Snapchat, Facebook, YouTube and X (formerly Twitter) are all slowly evolving to become the same infinite scrollable feed of
algorithm-optimized short videos from top performers.

And it turns out that I’m not the only one who has noticed that social networks have been slowly drifting away from their original purpose.
</item 7>

<item 8>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
That’s one takeaway, at least, from the recent shutdown of Pebble (formerly T2), a Twitter clone that put trust and safety at the forefront of its development
process. The app never gained more than 20,000 users, and the founders now believe that its approach to moderation may have been correct, but it wasn’t a growth
driver. In other words, people may say they want to feel safer when participating online, but clearly, that isn’t their only need.

Mozilla’s instance, still in private testing, has a small team of moderators that will grow when the instance launches to the public in the coming months.

“It’s not just a commitment, it’s beyond foundational for us — that is the thing that is going to be true about this instance,” O’Hara says of Mozilla’s plans
to heavily moderate its space. Though Mozilla will also rely on technology to do some of the work, it wants to make sure the human moderators, “feel protected,
supported and respected for their work,” she adds.
</item 8>

<item 9>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
If anything, Ohanian credits Twitter (now called X) with coming up with the best system for moderation so far.

“Community Notes, I personally think is a good — it’s the best measure I’ve seen yet. And frankly, I wish we had thought of it first,” he said.

Still, he called out the notion that a platform, like X, can be a town square.

“The public square [analogy]…falls flat. Because if that then implies that it’s a public square, which means like, what? a town or a city? If you’re the person
in charge of it, you’re the mayor. But there’s no democratic election,” Ohanian pointed out. “You’re a CEO. So then you’re like the king? If you want to lean
into the town square thing, fine. Just own it and say, yeah, it’s a town square and I am the infinite dictator, tyrant of it.”
</item 9>

<item 10>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
Threads, Instagram’s supposed “Twitter killer,” got 100 million sign-ups within five days; the decentralized platform Mastodon has 1.8 million monthly active
users; and the smaller upstart Bluesky hit 1 million users in September. Though still in beta, Spill’s community of 200,000 users may seem small in comparison,
but Spill’s rise has shown that quality matters just as much as quantity — that building a strong core community is just as valuable as high-speed growth, even
if the traditional venture model may not always incentivize thoughtful product development.

“If you have a great experience, you’re going to grow,” Terrell said. “Especially in a world where you have so many networks — there are all these choices that
people conceivably have — you’re gonna go to the one where you have the greatest experience and want to invest your time.”
</item 10>

<item 11>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The founders don’t regret their experience building Pebble, even though it didn’t turn out as they hoped, they said.

Says Cselle, “One thing that I learned is that there’s absolutely an audience that wants to see a new kind of Twitter-like platform built and will ask for the
features that Twitter has.” He suspects the team might stay together to work on something new after Pebble wraps, taking the learnings from Pebble with them.
That includes better communicating to users what great content should look like on a platform, and how they can win.

“We’re at an inflection point in social media,” says Oh. “We had one hypothesis with T2 and Pebble. It obviously didn’t pan out the way that we wanted it to.
But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
</item 11>

<item 12>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 12>

<item 13>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“We were really seized by people saying they wanted something that felt safer, where they could speak openly without dogpiling,” says Oh. “And so we were
confident that was an important thing to hit. Are there other things that are equally important and a great user experience? Yes.”

“Twitter is you have real people…saying real things,” Oh continues. “At the expense of using a word that’s overused, ‘authentic.’ And our theory was always if
you can create a space where people really feel like they can let down their hair, then you would get that authentic content. We didn’t quite get there,” she
admits.

Cselle agrees, noting that, perhaps, Pebble should have opened up enough space for disagreement to happen, while still drawing a hard line on the most
disagreeable parts of running a Twitter-like platform.

“I think we could have maybe moved a little bit further towards that,” he says. “There’s a lot of nice things that are said on Pebble, but maybe not enough
breaking news.
</item 13>

<item 14>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“They’re in the same ballpark, the sentiment is the same,” Barrett said. “Republican lawmakers in each state believe that — and said explicitly in the course of
debating and passing these laws — that ‘Silicon Valley oligarchs’… are ideological liberals, and they are censoring people in our states who are conservative,
and we are hereby ordering them to stop doing that.”

How could the Supreme Court decision affect social media companies?

If the court finds that social media companies don’t have a First Amendment right to curate the kinds of content that they allow, social platforms could look
very different, at least in states that are trying to limit their moderation powers. After years of slow progress on misinformation — and worrisome backsliding
on platforms like Elon Musk’s X — the Supreme Court’s decision could upend that process, sowing chaos online in the process.
</item 14>

<item 15>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
And if we can solve it, it will do so much good. It really, really will because we will find out we have more in common than not,” he said.
</item 15>

<item 16>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
It’s just that in a fediverse-dominated world, the way to win is not to achieve excellent lock-in and network effects. The only way to win is to build the best
product.

This is really not a particularly hot take, by the way. Even the most successful centralized platforms have long understood that a protocol-driven social web is
a good idea. Jack Dorsey used to say that Twitter was better as a protocol than a platform, and started the project that became Bluesky before also helping get
Nostr going. (We really don’t need to get into the whole story of what happened to Twitter since then, except to say that the speed with which that platform
changed made a lot of people acutely aware that we need a social ecosystem that can resist the whims of a single company or CEO.)

Meanwhile, Meta CEO Mark Zuckerberg seems to earnestly believe that bringing Threads to the fediverse is a good idea, both for its product and for its business.
</item 16>

<item 17>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
February 2023

Twitter

Announced on February 26, the company laid off more than 200 employees, including Esther Crawford, Haraldur Thorleifsson and Leah Culver. Since Musk took over
Twitter in October last year, the company’s headcount has fallen by more than 70%.

Poshmark

Announced February 24, Poshmark confirmed with TechCrunch that less than 2% of its workforce was affected, primarily in the U.S. The company employs roughly
800+ employees.

Green Labs

We do not have an exact figure of how many Green Labs plans to lay off its staff. Green Labs confirmed to TechCrunch that it is conducting a round of layoffs
that could impact at least 50% of its workforce.

Chipper Cash

Announced on February 20, the African cross-border payments platform conducted a second round of layoffs just 10 weeks after it cut approximately 12.5% of its
workforce. Chipper Cash relieved almost one-third of its workforce, about 100 employees.
</item 17>

<item 18>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
Another app that spun out from laid-off former Twitter employees, Pebble (formerly T2), recently shut down after a year, growing to just 20,000 registered
users. Its CEO, Gabor Cselle, previously had exits to Google and Twitter, but accepted that Pebble simply wasn’t competitive enough to stay afloat. But while
Pebble sought to essentially clone Twitter, Spill’s team is building something that looks very little like it.

“People are looking for something new,” Terrell said. “I think things that have really defined, unique value propositions are going to win over the long term —
it might not be like there’s one winner-take-all.”

Earlier this year, TechCrunch covered the founding of Spill, piquing the interest of Kapor Capital, which ended up writing the company its first check.

“We had over 25,000 people join our waiting list in less than 24 hours from that article,” Terrell said.
</item 18>

<item 19>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
A social social network network: Lauren reports that a former TikTok employee is building a social app for content creators to network and “spill the tea,” so
creators can help each other out making better, more engaging content.

X may go ad-free?: It appears that X (formerly Twitter) is testing three tiers of its Premium service, its CEO says. Under the hood, code shows one tier may be
ad-free. If it’s also troll free, please take my money right now.

Top reads on TechCrunch this week

A lot of amazing news on the site this week, but if we go by the raw numbers, here are the most popular stories — the ones that I didn’t already cover above,
that is.

Passwords? We don’t need no steenkin’ passwords: Passkeys are a phishing-resistant alternative to passwords that allow users to sign in to accounts using the
same biometrics or PINs they use to unlock their devices or with a physical security key. Google is now making it the default sign-in method for all users.
</item 19>

<item 20>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The grip Twitter, now called X, has on the market may be stronger than some believed. Unfortunately, that’s led to the first casualty among Twitter
alternatives, as the startup Pebble (formerly T2), is shutting down. The would-be X rival had grown a small but engaged community on its microblogging service
that aimed to dupe Twitter’s features, from its verification systems to functionality like DMs. But the company says they’ve run out of time to make Pebble
happen — the app maxed out at 3,000 daily active users, out of 20,000 registered users. That daily user figure fell to 1,000 daily users following its
rebranding from T2.

In part, the sizable competition coming from other Twitter rivals was to blame, says co-founder and CEO Gabor Cselle, who had exits to Google and Twitter prior
to starting Pebble.

“I think the competitive landscape evolved faster than we had thought,” he explains.
</item 20>

<item 21>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Though he’s glad that access to the technology has broadened, that has come with many complications.

“I don’t like the idea that the internet I’m so nostalgic for was actually an ivory tower,” he said, but on the flip side, he said, we’re all exposed to more
people, in real-time, than we could have ever fathomed. And that can put people at odds with one another and with everyone’s different versions of the truth —
even if their truth is believing in something as far-fetched as the earth being flat, he said, pointing to one example that has a following on Reddit.

Meanwhile, asking platforms like Facebook, Twitter, or Reddit to be in charge of verifying truth and accuracy is a difficult task, he said. Ohanian also called
out TikTok as “an arm for the Chinese government” that’s really good at “feeding us what we want, when we want it” — in other words, not always a fountain of
truth, either.
</item 21>

<item 22>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Does Reddit co-founder Alexis Ohanian have regrets over his creation? The entrepreneur and founder of venture fund Seven Seven Six, speaking at an event at the
University of Virginia, espoused a fairly negative view of the current social media landscape and the creator economy — an environment where he says, everyone
has “been so trained and conditioned now to just get the likes and retweets, get the upvotes…I wish I could be more optimistic about it, but we’re seeing it
play out before our eyes.” Still, he ultimately described himself as a “techno-optimist” — an apparent nod to a16z General Partner Marc Andreessen’s recent
diatribe where he claimed technology could solve all our problems.

“Guilty as charged,” Ohanian said, during his interview with Kara Swisher. “I do think that pendulum will swing back. I think even swifter than we realize.”
</item 22>

<item 23>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.
</item 23>

<item 24>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
“What are the implications of federation in the front end? How do we think about curation and all the things, all the capabilities and tools that we’ve created
over the years? How does that work in a world that’s federated, and from a user experience point of view? That is a great question,” he said.

Despite all the changes, Flipboard is not in need of raising funds to support its new developments. It’s running off the profits of its own business as it moves
in this direction.

The company is also betting on the fact that federated social media may only be the beginning of what’s to come for the web overall.

“I saw what was happening with ActivityPub and it became very clear to me that this is the future of the web, period,” McCue said. “The social web is people
linking to pages and people linking to people. So it’s a much more intricate web.”

He sees Flipboard as a part of that opportunity.
</item 24>

<item 25>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
I’d bet heavily that ActivityPub becomes the default choice over time, but ultimately, it doesn’t matter so much which protocol wins as long as one of them
does. We don’t need two internets, and we don’t need two social protocols. We need one thing that is both simple enough and big enough to handle all the ways we
connect with each other online. No centralized platform has ever been big enough. The fediverse can be.

Mastodon is the biggest thing in the fediverse so far, but it may not be for long. Image: Mastodon

Decentralizing social media can sound like a sort of kumbaya anti-capitalist manifesto: “It’s about openness and sharing, not capitalism, man!” In practice it’s
the opposite: it’s a truly free market approach to social networking. Mastodon may not be interested in becoming a trillion-dollar company, but there’s no
reason there can’t be plenty of those built on the fediverse.
</item 25>

<item 26>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“Thanks to the First Amendment, Florida and Texas cannot force websites or social media apps to host hateful content, misinformation and spam, as their deeply
misguided laws would require,” Oregon Senator Ron Wyden, who co-authored Section 230, a law that protects social media companies’ content moderation decisions,
told TechCrunch. “… A ruling in favor of the Texas and Florida laws would create utter chaos, and make many sites worthless to regular users who want to watch a
funny video or see family photos.”

NetChoice President Steve DelBianco also warned that allowing the state laws to go into effect would unleash “a tidal wave of offensive content and hate speech
crashing onto users, creators, and advertisers” that would force Americans to wade through “racial epithets, aggressive homophobia, pornographic material,
beheadings, or other gruesome content” just to use social apps.
</item 26>

<item 27>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
On Equity this week, the crew talked about the SBF trial; deals from VR firms Rainforest, At One Ventures, Section 32 and Greylock, where venture funding has
declined; and how Fearless Fund, a firm founded to invest in women of color, is being barred from awarding grants to Black women founders.

Meanwhile, Found featured Esther Rodriguez-Villegas from Acurable, a medical device company that makes patient-friendly wearable devices to diagnose and manage
respiratory conditions at home. As a career-long academic, Rodriguez-Villegas talks about how she never intended to be a founder until she learned about how the
currently available medical devices make it extremely difficult to detect and treat diseases like sleep apnea and epilepsy.

And over on Chain Reaction, Jacquelyn did a crossover episode with Alex about the SBF trial.
</item 27>

<item 28>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 28>

<item 29>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Hippo

Plans to eliminate 120 roles, the company said in an SEC filing October 30.

Liberty Mutual

Plans to cut 850 members of its staff, the insurance company confirmed October 27.

Salsify

Cut 110 jobs and are moving some roles abroad, CEO and co-founder Jason Purcell announced October 25.

Slync

Is shutting down, months after its former CEO Chris Kirchner was arrested on fraud charges and sued the company to pay for his legal bills.

Shipt

Announced October 24 in a company statement that it laid off 3.5% of their workforce and will close many open roles.

Pebble

The X alternative previously known as T2 announced October 24 that it is shutting down its operations.

Nokia

Announced October 19 that it will cut up to 14,000 jobs, or 16% of its workforce, following a sizable dip in Q3.

Roblox China

Cut 15 roles on October 24, two years after the joint effort with Tencent abruptly paused its service.
</item 29>

<item 30>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think the competitive landscape evolved faster than we had thought,” he explains. “I didn’t think that quite as many people — established organizations and
newcomers — would try to do the same thing that we were doing and in very similar ways.”

Today, the market for Twitter alternatives is a crowded one, to say the least. There are numerous services for consumers to try, including the open source-based
platform Mastodon, a soon-to-be decentralized system from Bluesky, plus smaller startups like Spill, Spoutible and Post, as well as a new app from Meta called
Instagram Threads.

Then, of course, there’s the X factor to consider. The default platform for short posts retains some pull, despite — or perhaps because of — new owner Elon
Musk’s antics. Though the company may not have yet succeeded from a business and financial perspective, it’s been hard for others to duplicate its function as a
breaking news platform and place for spirited debates.
</item 30>

<item 31>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
There’s this picture from more than a decade ago that still goes viral on the web every once in a while. You’ve probably seen it: it was created by the venture
capitalist Andrew Parker, and it compares a few dozen startups with subsections of Craigslist. Back then, Craigslist was all things to all people online, and a
generation of startups figured they could do part of the job a lot better.

Some of those companies failed — sometimes because of Craigslist! — but some also became gigantically successful. Airbnb worked because it was more searchable,
reliable, and trustworthy than a random Craigslist listing; StubHub sold you secondhand tickets without the 50 percent chance those tickets were fake; Etsy
offered a much more fun shopping and discovery experience than a bunch of text listings and crappy photos.

For a decade, startups tried to unbundle Craigslist. Now they can do the same for social networks.
</item 31>

<item 32>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think Threads and others have encountered this as well,” adds Pebble co-founder and CTO Michael Greer, previously Discord engineering head. “The durability
of the network effect of Twitter is stronger than anticipated,” he says.

Despite these challenges, Pebble initially had decent retention. By week four, it was seeing retention rates of 30%, in fact. Its invite list was also working,
as 60% of people who received a Pebble invite converted to become users. In addition, some 10,000 users arrived at that list from early press, like TechCrunch’s
coverage of its first outside funding — a $1.1 million angel round that included investors like former Google VP Bradley Horowitz, Android co-founder Rich Miner
and the former CEO of Wikipedia, Katherine Maher. Data from Similarweb indicates the Pebble website topped 99,300 visits last month, indicating repeat usage.

The company’s hypothesis was that consumers were hungry for a Twitter alternative that prioritized trust, safety and moderation from the start.
</item 32>

<item 33>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
But as for social media as a whole, of which Reddit would presumably play a part, Ohanian said it brings out the “worst parts of our nature in many ways.”
What’s more, he expressed worries about the desire for people to be first and win on whatever leaderboard a platform features, which then leads to a situation
where misinformation is able to spread.

Humans evolved trusting the faces & voices we knew around the campfire. The idea of believing something bc of experts/institutions you've never met is actually
new to our species. I visited @UVADemocracy + chatted with @karaswisher + @dkroy about what's next for social media. pic.twitter.com/PMBXi7T3vv — Alexis Ohanian
🇦🇲 (@alexisohanian) October 31, 2023

“Now, there’s so much pressure to be first, and UGC — user-generated content — will always win. It won’t always be right, though.
</item 33>

<item 34>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
Given that Mastodon today has around 1.5 million monthly active users, this could be a notable bump for the fediverse when Flipboard’s integration fully rolls
out, assuming Flipboard’s users don’t opt out.

Flipboard is only one of now several companies that has embraced decentralized social media. In addition to X rival Instagram Threads, which began testing
ActivityPub last week, other tech companies are moving in this direction, as well. Automattic made it possible for all WordPress.org and WordPress.com blogs to
become federated, and said it’s working on doing the same with Tumblr next year. Medium and Mozilla have also set up their own servers, and the latter backed a
Mastodon client called Mammoth, too.

For Flipboard, after integrating its back end with the fediverse, the company may reconsider what its front end should look like, too, for this new age of
social media.

“The front end was built at a time pre-federation,” noted McCue. “What are the implications of federation in the front end?
</item 34>

<item 35>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
Consumers are hungry for a new way of social networking, where trust and safety are paramount and power isn’t centralized with a Big Tech CEO in charge… or at
least that’s what Mozilla believes. The mission-driven tech company behind the Firefox browser, Pocket reader and other apps is now investing its energy into
the so-called “fediverse” — a collection of decentralized social networking applications, like Mastodon, that communicate with one another over the ActivityPub
protocol.

The idea is to rethink social networking from the ground up.

Today, social networks are often run by large corporations — like Meta, Snap and Google — where advertisers pay the bills. This has created a world where modern
social networks are profit-driven, and consumer needs aren’t always at the forefront, Mozilla believes. This disconnect between what people want and what
today’s networks offer hit an inflection point last year, when billionaire Elon Musk bought Twitter, leading to a surge of interest in alternatives to
centralized social networking, including apps like Mastodon and Bluesky.
</item 35>

<item 36>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
“Suddenly, we’re supporting a city, we have data costs, we have infrastructure, we have to accelerate our building times, so we really needed an extension to do
that effectively.”

His journey with Spill started last year at AfroTech. He arrived at the conference just days after being laid off from what was then Twitter, along with about
3,000 others amid Elon Musk’s acquisition. Terrell hopped on a call with former Twitter colleague DeVaris Brown about a business idea and spent three days in an
Airbnb creating the pitch deck for what became Spill. At AfroTech, he went around taking last-minute coffee meetings, seeing who would give his company a
chance. Flash-forward to this year and Spill is the cool kid in town.

So many apps have sprung up over the last year in an attempt to become “the new Twitter,” but no app has emerged as the dominant refuge for disillusioned
Twitter users.
</item 36>

<item 37>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Critics on both the right and the left described the launch as a disaster.

Conservative organizations have proven especially out of their depth when facing the wrath of extremely online social media users, who mobilize trolling for
social justice. In 2020, TikTok users reserved hundreds of tickets for a Trump rally and never showed up, leaving the president to face swaths of empty seats.
When a Texas anti-abortion group opened an anonymous tip form to enforce the state’s ban on abortion after six weeks, Gen Z activists flooded the site with
Shrek porn. This year, TikTok and Twitter users shut down the Missouri attorney general’s tip form for reporting clinics that provide gender-affirming care.
Within days of launching, the site was spammed with fanfiction, satirical anecdotes about kids getting “transed” and fanfiction. The attorney general’s press
secretary blamed the site’s downfall on activists “hacking the system.”
</item 37>

<item 38>
X plans to use payment, phone and ID verification to stop bots, in addition to the new $1/yr fee (TechCrunch, 2023-10-18T17:33:48+00:00)
A report this week from market intelligence firm Similarweb, for example, found that X’s web traffic and monthly active users on mobile had declined
post-acquisition.

Users today have far more options for microblogging and public discourse, thanks to products from Twitter-like startups and tech companies such as Spill,
Bluesky, Pebble, Countersocial, Spoutible, Hive and the open source platform Mastodon, as well as efforts from tech giants, like Meta’s new app Threads. Few of
the other social apps currently charge a fee, unless it’s for premium features, like some of the third-party Mastodon apps do.

In addition to presenting friction at sign-up, other critics have argued a fee could exclude genuine users and increase the digital divide.
</item 38>

<item 39>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
“The [World Health Organization] now calls it the loneliness epidemic. And if they say that it’s an epidemic, it’s because it’s actually infectious. In other
words, if you’re isolated, your loved ones are too because you’re unreachable. So during the two hours you’re on TikTok, they have no one to talk to,” Martin
told me.

“And at the same time, the human needs that the social consumer space can fulfill are no longer covered by these products, whereas they used to be,” he added.
“In the early days of Facebook, I don’t know if you remember, profiles were sort of a mash-up. There were drawings, games, photos, text. You’d write long
comments, it could be a poem . . . And on the other hand, it was a reminder that you mattered to these people.”

According to him, the current generation of social networks are very passive.
</item 39>

<item 40>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
– Jonathan

Signing off

This weekend, pour one out for Tumblr, one of the most interesting social networks on the internet, which appears to be in trouble. It’s not dying, but it’s
not… not dying, you know? Whatever happens next, this state of affairs is a bummer for a lot of reasons, including that Tumblr promised to work with ActivityPub
— which would have been a big win for the fediverse — and I think is still maybe the web’s best and most versatile posting tool. Matt Mullenweg, the CEO of
Automattic (which owns Tumblr), spent a bunch of time this week answering people’s questions about the future of Tumblr, and it paints a sad but interesting
picture of what it really takes to build a better social network. It’s all making me root for Mastodon even harder.
</item 40>

<item 41>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
For Pebble, that’s certainly true. It was a perfect storm of competition, X’s continued traction, the lack of a native app, a brand that didn’t resonate and a
space that was maybe a little too safe to be as addictive and as fun as the original.

With Pebble’s winddown, being announced to users today at 9 AM PT, early adopters will have the option of exporting their Pebble archive as a zip file that
loads a mini web page showcasing all their old posts. Pebble won’t be directing users back to X however, or any other social network.

As for where the founders will now hang out?

“I don’t know. Maybe LinkedIn,” Cselle jokes.

Pebble is also returning a small amount of the funds left over to its investors, more as a courtesy to show financial responsibility. The Pebble website will be
shut down on November 1st and there aren’t further plans for the IP at this time.
</item 41>

<item 42>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 42>

<item 43>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Third-party Reddit app Narwhal says it hopes to survive Reddit’s app purge with a subscription plan.

Apropos “getting together,” Amanda reports that VidCon is still kickin’. For the first time, the conference hosted an industry leadership summit, where creator
economy experts and creators could hash out their grievances with the state of the business and share ideas to make the job of a creator more sustainable. That
makes sense, unlike creators raising venture capital: It’s so eye-wateringly hard to make money as a creator, I’d love to see the pitch that convinces a VC to
cut a check to a creator, with a realistic expectation of a venture-scale return.

Maybe they were hiding behind the sofa?: Sarah reports that Mastodon actually has 407K+ more monthly users than it thought — and it seems like Twitter has a lot
more traffic than Musk said. Still, the peak now is about the same as it was a decade ago, and it’s unclear what the social platform can do to encourage more
growth.
</item 43>

<item 44>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
I think even swifter than we realize.” The war in the Middle East may even be the tipping point, he theorized, as it’s now “abundantly clear” that we need
something better to figure out the world than the platforms we have today.

The founder, as you may recall, notably resigned from Reddit’s board in 2020 over differences of opinion on moderation and wanting to crack down on communities
about violence and hate. It was a moment where he seemed to be reckoning with what his platform had wrought. At the time, he asked for his board seat to be
replaced by a Black board member. The company agreed, appointing its first Black board member with Y Combinator partner Michael Seibel.

Reddit also later banned several toxic communities, including r/donaldtrup following violence at the U.S. Capitol, and Ohanian says that the site has “gotten
demonstrably better,” since, as well as better as a business.
</item 44>

<item 45>
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
</item 45>

<item 46>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
There are already a few platforms built on ActivityPub and embracing the ideas of the fediverse. Apps like Mammoth and Ivory are showing the potential for
different user experiences on top of the same data and infrastructure. But so far we’re mostly in the “popular app, but federated” phase of this transition.
(Which is definitely better than “popular app, but blockchain” from a couple of years ago… but not by much.)

So far we’re mostly in the “popular app, but federated” phase of this transition

Mastodon deliberately looks and feels like Twitter. (So do Firefish and Pleroma and GoToSocial and others. The fediverse is super into replacing Twitter.)
Pixelfed is Instagram through and through. Lemmy’s features are virtually all ripped straight from Reddit. Almost everything in the fediverse is a one-to-one
competitor to an existing platform: PeerTube to YouTube; Friendica to Facebook; BookWyrm to Goodreads; on and on it goes.
</item 46>

<item 47>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
Not enough conversation.”

There were other things that may have hurt Pebble’s traction, too, including its lack of a native mobile app. As T2, the startup prioritized building for the
web to be agile and more flexible, but lost out on the opportunities for discovery driven by the app stores. Notes Greer, “that might have been a mistake.”

In retrospect, Cselle also questions if the rebranding from T2 to Pebble may have been an issue.

“I’m wondering if the Pebble rebrand had actually softened the value prop[position] because T2 just encapsulated it so much — that it was really duplicative [of
Twitter],” he says.

Like most startup shutdowns, the answer as to “why” is not any one thing but a combination of factors. For Pebble, that’s certainly true.
</item 47>

<item 48>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
“Not everyone wants to use one product,” he said to my colleague Alex Heath earlier this fall, “and I think making it so that they can use an alternative but
can still interact with people on the network will make it so that that product also is more valuable.” Mark Zuckerberg! Said that! He understands that people
want to feel like their social connections and content belong to them, not to a company that can disappear or pivot or change its ways. And let’s be real: if
Meta can’t build and maintain the One True Social Network For Everybody, nobody can. It’s well past time to try something else.

The only way to see your Facebook posts is to sign up for Facebook’s rules, Facebook’s aesthetic, Facebook’s features, Facebook’s moderation, Facebook’s
algorithm, and Facebook’s business model.

Right now, every social platform is a universe unto itself.
</item 48>

<item 49>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Not Sonos fast there: Audio company Sonos scored a big $32.5 million win against Google a while back. Now it transpires that the company’s patents were deemed
unenforceable and invalid. Whoops. A federal judge threw out the $32.5 million win this week.

When it clicks, it really clicks: We took a deep look at Pixel 8, and our team discovered that it delivers solid performance and design upgrades. The camera got
a particularly enthusiastic shout-out, with Darrell declaring that Google’s Pixel 8 Pro camera is the new mobile photography champ, and Brian waxing lyrical
about the phone’s picture-snapper in “The camera’s still the thing.”

Who needs computers anyway?: It seems like all our mobile devices may be starting to cannibalize sales of personal computers — Ron reports that PC shipments
decline slows in Q3 2023 and that Apple’s sales plunge over 23%.

Let’s get together

Reddit’s API-powered chaos continues, but it appears that things are starting to resolve a little.
</item 49>

<item 50>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 50>
=== END ITEM id=271739166f813f64 ===

=== ITEM id=bd23a71d44a81caa ===
Gold sentences and facts:
Sentence 0: Postponements of such reports could complicate things for the Fed, which has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.
  Fact 0.0: Postponements of such reports could complicate things for the Fed.
  Fact 0.1: The Fed has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.
Sentence 1: With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest rates and could soon turn to cutting them.
  Fact 1.0: Inflation is down from its peak two summers ago.
  Fact 1.1: Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest rates.
  Fact 1.2: Wall Street is hopeful that the Federal Reserve could soon turn to cutting interest rates.

Retrieved items:
<item 1>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Tweet of the day

Quote of the day “Our view is that in the high likelihood there is another cash rate increase on Tuesday, that is going to be the nail in the coffin for
consumer confidence,” said CreditorWatch chief economist Anneke Thompson, amid warnings from some economists that another rate rise, as some market watchers are
predicting, will dampen spending intentions just in time for Christmas. You may have missed The men who built pubs and liquor juggernaut Endeavour Group into
the world’s largest poker machine operator have triggered a public spat over the future of a business that they say has lost its way.
</item 1>

<item 2>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Fed put such pressure on the economy intentionally, hoping to starve inflation of its fuel. It wants the job market to cool, particularly pay raises going
to workers. The Fed fears too-strong pay gains could create a vicious cycle that keeps inflation high.

Analysts said Friday’s jobs report offered encouraging signals for the Fed, with average hourly earnings rising less in October from September than expected,
though it doesn’t mean the job is done. Treasury yields in the bond market tumbled immediately after the jobs report, releasing more of the pressure that had
built up on Wall Street. The yield on the 10-year Treasury eased to 4.52 per cent from 4.67 per cent late Thursday and from more than five per cent last week,
when it hit its highest level since 2007. In the US, traders are moving up expectations for when the first cut to interest rates by the Fed could happen,
potentially by the summer, according to data from CME Group. Such cuts can act like steroids for financial markets.
</item 2>

<item 3>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.
</item 3>

<item 4>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The laggards On the losing end, energy companies (down one per cent) weighed down the index. Woodside (down 0.8 per cent), Santos (down 0.4 per cent) and Ampol
(down 1.2 per cent) dropped, along with coal miners Yancoal (down 1.7 per cent) and Whitehaven (down 2 per cent). Lynas Rare Earths (down 3.1 per cent) was the
biggest large-cap decliner, while industrials companies (down 0.2 per cent) including Computershare (down 2.9 per cent), Seven West Holdings (down 1.9 per cent)
and Worley (down 1.3 per cent) were also weaker. The lowdown

TMS Capital portfolio manager Ben Clark said the big news for markets on Monday was the trading update from Westpac, which he said led a resurgence in the
banking sector’s performance.
</item 4>

<item 5>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
The Australian dollar was US64.20¢ at the close as the US dollar fell for a fifth straight day, its longest losing streak since July. ‘Everybody has one eye on
the Middle East conflict and one eye on what’s happening with bond yields.’ John Praveen, managing director at Paleo Leon The 10-year Treasury yield came off
its 16-year peak, on track for its steepest single-day drop since May, as trading resumed in the US bond market, which had been closed for a holiday on Monday.
Ten-year yields tumbled as much as 18 basis points to 4.62 per cent. Fed swaps currently show about a 60 per cent chance the Fed will stay on hold in December,
compared with 60 per cent odds on another interest rate rise by then, just a week ago. But while oil prices eased after Monday’s rally, investors were still
closely monitoring developments the Middle East.

Israeli air strikes attacked Gaza on Tuesday, razing entire districts in the densely populated and impoverished enclave.
</item 5>

<item 6>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
They also continued to decelerate from earlier months. A second economic report from the morning likewise offered both encouragement and caution for financial
markets. It said slightly fewer US workers than expected applied for unemployment benefits last week. On one hand, it indicates a job market with few lay-offs,
which means strength for the overall economy. But it could also be adding fuel to keep upward pressure on inflation.

After the reports, the 10-year Treasury yield rose to 4.7 per cent from 4.56 per cent late Wednesday. The two-year Treasury yield, which more closely tracks
expectations for the Fed, climbed to 5.06 per cent from 4.99 per cent. Yields are still down slightly for the week so far. After jumping last week to their
highest levels in more than a decade, yields regressed following speeches that investors saw as hints that the Fed may not raise its main overnight interest
rate further.
</item 6>

<item 7>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The stock market has largely been taking its cue from the bond market recently, and weak results announced in the afternoon for an auction of 30-year Treasury
bonds sent yields of all kinds higher. Higher yields can knock down prices for stocks and slow the economy by making borrowing more expensive. Yields had
already been on the rise in the morning after a report that showed inflation at the consumer level was a touch higher last month than economists expected. That
raises worries about the Federal Reserve keeping its main interest rate high for a long time, as it tries to undercut inflation by knocking down investment
prices and slowing the economy. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it comes to what it will do on November 1.
The inflation report, though, also had some encouraging nuggets for financial markets underneath the surface. After ignoring prices for food and fuel, which Fed
officials see as a better predictor of where inflation is heading, prices that consumers had to pay last month were in line with expectations.
</item 7>

<item 8>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
“Westpac’s results were better than expected, especially because it’s been a bit of a perennial underperformer in the past few years,” he said, adding that it
set the tone for results from NAB and ANZ, which are due to report in the next week. Loading Clark also said there was a reversal of the interest-rate trade
seen in recent weeks. “Long-term bond yields came off quite significantly in the last few days following weaker than expected employment data on Friday, which
has seen some investors swivel back into growth stocks,” he said. The market has now switched from pricing in an 80 per cent chance of a rate rise on Tuesday to
50-50, Clark said.

On Wall Street, the S&P 500 climbed 40.56 points on Friday, or 0.9 per cent, to 4,358.3.
</item 8>

<item 9>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
It’s again near its highest level since 2007.

Loading Treasury bonds are seen as some of the safest investments possible, and when they pay higher yields, investors are less likely to pay high prices for
stocks and other riskier investments. That’s a big reason why the S&P 500 dropped 4.9 per cent in September to drag what had been a big gain for the year down
to 11.7 per cent. Treasury yields have been climbing sharply as Wall Street accepts a new normal where the Federal Reserve is likely to keep interest rates high
for longer. The Fed is trying to push still-high inflation down to its target, and its main tool of high interest rates does that by trying to slow the economy
and hurting prices for investments. The Fed’s main interest rate is at its highest level since 2001, and the central bank indicated last week it may cut
interest rates next year by less than it earlier expected.
</item 9>

<item 10>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
It had been above 5 per cent, and at its highest level in more than a decade, during October. The yield on the two-year Treasury, which more closely tracks
expectations for the Fed, went on a jagged run following the economic reports. It fell from 4.61 per cent just before the reports’ release to 4.57 per cent and
then yo-yoed before easing back to 4.57 per cent.

Loading Traders widely expect the Federal Reserve to hold its key interest rate steady at its meeting next week, before potentially cutting rates in March,
according to data from CME Group. Fed officials have recently hinted that the federal funds rate may indeed already be at its peak. It’s above 5.25 per cent, up
from nearly zero early last year. But Fed Chair Jerome Powell and others have also warned Wall Street about being overzealous in its predictions about how early
a cut could happen. Lower yields have been one reason prices for cryptocurrencies have been rising recently.
</item 10>

<item 11>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Infratil (down 3.6 per cent), Meridian Energy (down 2.7 per cent) and Resmed (down 2 per cent) were among the biggest large-cap decliners. The lowdown The
Australian sharemarket shook off a positive lead from Wall Street on Wednesday after hotter-than-expected inflation data raised the prospect of another interest
rate rise at the Reserve Bank’s next meeting, which would weigh on the equity market. Interest-rate sensitive sectors, including REITS and consumer companies,
were among the weakest on the local bourse as investors digested the news.

GSFM investment strategist Stephen Miller said the inflation data, taken with the October Reserve Bank board meeting minutes, which revealed the board had “low
tolerance for a slower return of inflation to target than currently expected”, made the possibility of an interest rate rise from the central bank in November a
“near certainty”. However, commodity prices supported mining companies as iron ore prices increased 2.9 per cent overnight, bolstering Australia’s iron ore
giants.
</item 11>

<item 12>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Dow Jones Industrial Average gained 222.24, or 0.7 per cent, to 34,061.32, and the Nasdaq composite jumped 184.09, or 1.4 per cent, to 13,478.28. Stocks
surged through the week on rising hopes that the Federal Reserve is finally done with its market-crunching hikes to interest rates, in order to get inflation
under control. A report on Friday underscored that pressure is easing on inflation after it showed employers hired fewer workers last month than economists
expected. It’s a stunning turnaround from just a week ago, when Wall Street was reeling after the S&P 500 had fallen 10 per cent below its high point for the
year. That sent Wall Street’s main index into what investors call a “correction.” Since then strong profit reports helped drive some stocks to towering gains.
Generac, a maker of back-up generators, soared nearly 28 per cent for its best week since its stock began trading in 2010.
</item 12>

<item 13>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Such cuts can act like steroids for financial markets. For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the
only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report. A separate report on Friday said growth
in US services industries, such as finance and construction, was weaker last month than economists expected. The report from the Institute for Supply Management
also suggested a slight easing in prices.

Excitement about a potentially easier Fed was more than enough to offset a fall for Apple, which is Wall Street’s most influential stock. The most valuable US
stock fell 0.5 per cent despite reporting stronger profit for the latest quarter than analysts expected. Analysts said investors were likely disappointed with
Apple’s forecast for revenue for the last three months of 2023.
</item 13>

<item 14>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“The bond market is retracing,” he said. “There’s also been strength in resource stocks amid rumours that the Chinese government might launch a substantial
stimulus package.” Clark said interest-rate sensitive sectors climbed on increased probabilities of an easing in interest rates. “Consumer discretionary and
information technology companies are winners in lower-interest rate environments,” he said. As more annual general meetings kick off in the next few weeks,
Clark said he would be looking at trading updates, especially of companies that didn’t give solid guidance in their last update.

Elsewhere, Wall Street advanced overnight, with the S&P 500 and Nasdaq touching three-week highs as dovish comments from the Fed officials pushed Treasury
yields lower. Following remarks from other top Fed officials on Monday, Atlanta Fed President Raphael Bostic said the US central bank did not need to raise
interest rates any further, and that he saw no recession ahead. Oil edged lower, with West Texas Intermediate settling below $US86 a barrel following its
biggest rally since April.
</item 14>

<item 15>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
That growing understanding has sent yields in the bond market to their highest levels in more than a decade, which in turn has undercut prices for stocks and
other investments. Treasury yields rose again on Tuesday following a mixed batch of reports on the economy. The yield on the 10-year Treasury edged up to 4.55
per cent from 4.54 per cent late on Monday and is near its highest level since 2007. It’s up sharply from about 3.5 per cent in May and from 0.5 per cent about
three years ago. One economic report on Tuesday showed confidence among consumers was weaker than economists expected. That’s concerning because strong spending
by US households has been a bulwark keeping the economy out of a long-predicted recession.

Besides high interest rates, a long list of other worries is also tugging at Wall Street. The most immediate is the threat of another US government shutdown as
Capitol Hill threatens a stalemate that could shut off federal services across the country.
</item 15>

<item 16>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Energy was among the laggards after its more than 3 per cent rally on Monday. A measure of US-listed Chinese shares added 3 per cent amid reports that the Asian
nation is considering new economic stimulus. “Policymakers have begun to acknowledge a lesser need for further policy action given financial conditions have
tightened considerably after the recent surge in Treasury yields,” said Ben Jeffery at BMO Capital Markets. “This acknowledgment may have reduced angst around
the need for additional rate increases.” Loading Investors will be watching for any hints in the September Fed meeting minutes due on Wednesday that would
suggest the Fed may not follow through with the last rise indicated in its economic projections, according to Anna Wong at Bloomberg Economics. Two critical
coming indicators for the US economy — Thursday’s consumer price index and Friday’s University of Michigan consumer-sentiment survey — may give a more
definitive read, she noted.
</item 16>

<item 17>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.
</item 17>

<item 18>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 18>

<item 19>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
That’s reviving questions about whether central banks can deliver price stability all on their own.

“If we were designing optimal policy arrangements from scratch, monetary and fiscal policy would both have a role in managing the economic cycle and inflation,
and that there would be close coordination,” Philip Lowe said in his last speech as Australian central bank governor in September.

Economists surveyed by Bloomberg predict fiscal policy will somewhat counteract the Fed’s efforts to rein in inflation in the US.

“It’s true that there are circumstances where working hand in hand and supporting each other has proved helpful,” ECB President Christine Lagarde told a panel
discussion in June at the institution’s annual economic forum.

Fed Chair Jerome Powell, who sat to her right, signaled he wasn’t ready to rely on that kind of cooperation. “Our assignment is to deliver price stability kind
of regardless of the stance of fiscal policy.”

Central bankers warn that any failure to scale back fiscal spending risks coming at the cost of yet higher interest rates.
</item 19>

<item 20>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.
</item 20>
=== END ITEM id=bd23a71d44a81caa ===

=== ITEM id=4e09a60f573969a8 ===
Gold sentences and facts:
Sentence 0: If the debt continues to rise more than the economy, risks will increase, and the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.
  Fact 0.0: If the debt continues to rise more than the economy, risks will increase.
  Fact 0.1: If the debt continues to rise more than the economy, the government may 'crowd out' private investment.
  Fact 0.2: The government may crowd out private investment as it competes with the private sector for funds to service its debt.
Sentence 1: And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63% and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr. told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).
  Fact 1.0: President Ferdinand Marcos, Jr. said 'We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63%'.
  Fact 1.1: President Ferdinand Marcos, Jr. said the Philippines' debt-to-GDP ratio of about 63% is a little high for the country and is not ideal.
  Fact 1.2: President Ferdinand Marcos, Jr. said this to members of the US-ASEAN Business Council who visited Malacañang.
  Fact 1.3: ABS-CBN reported this on Aug. 9, 2023.

Retrieved items:
<item 1>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
To this end, the first goal of the PDP is social transformation which includes, among others, sustained expenditure on social services, mainly education,
health, and social protection.

The first two goals of the PDP help attain the developmental objectives of efficiency and equity, which rest on the pillars of sustainability and resilience.
The third fosters societal resilience: an enabling environment encompassing institutions, macroeconomic stability, and the physical and natural environment.

Institutions are vital to economic acceleration, which is why the government’s steps to enhance the ease of doing business are most welcome. Infrastructure
development also enables an economy to sustain higher levels of growth, which, in turn, catalyzes yet more investment. The Public-Private Partnership (PPP) Act
is up for the signature of President Marcos and, if approved, is expected to further enhance the business atmosphere in mobilizing private resources for
infrastructure development.

The government aims to sustain its infrastructure program at 5-6% of GDP through 2028 amid a six-year medium-term fiscal framework.
</item 1>

<item 2>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
the government borrows money either from foreign sources or from the domestic capital market which increases the debt stock of the NG and its debt servicing
requirements; b.) the government borrows money from the Bangko Sentral ng Pilipinas; or, c.) the government withdraws funds from its cash balances in the
Treasury. Historically, National Government expenditures have always exceeded total revenues resulting in annual budget deficits, the DBM declared. “Thus, the
National Government had to resort to borrowing to cover said deficits which resulted in the ballooning of foreign and domestic debts,” the budget office
admitted.

“However, in 1994, the government broke the deficit trend by posting a budget surplus of P16 billion through an aggressive privatization and revenue generation
program and a prudent expenditure program. Since then, the government has been exerting efforts to maintain the surplus budget policy. The surplus budget policy
is important to encourage economic growth,” the DBM said.
</item 2>

<item 3>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
This plan gradually narrows the deficit to 3% of GDP by 2028, down from 7.3% in 2022, such that the debt-to-GDP ratio is reduced from nearly 61% in 2022 to a
more sustainable level of at most 53% by 2028.

DEBT MANAGEMENT

The National Government’s debt was less than 40% before the pandemic struck. It expended much of its fiscal space combatting the pandemic, incurring debt and
large budget deficits. While the current 61% debt-to-GDP ratio may be manageable for an emerging economy like the Philippines for some time, the country may not
have the fiscal space to respond to another potential domestic or external shock. If the debt continues to rise more than the economy, risks will increase, and
the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.

The planned fiscal consolidation entails harmonizing the revenue needs with the promotion of investment through the structure and administration of the reformed
incentive system.
</item 3>

<item 4>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The National Government plans to raise more revenue to finance the country’s socio-economic needs, largely through a progressive and simplified tax system, more
efficient and effective tax collection measures through digitalization and, to some extent, from policy measures such as value-added tax on digital service
providers and excise tax on sweetened beverages and junk food.

Achieving fiscal stability presupposes the sustainability of fiscal policy, and fiscal stability helps “crowd in” private investment.

PRICE STABILITY AND INFLATION

In his second State of the Nation Address (SONA), President Ferdinand Marcos, Jr. singled out inflation as the country’s most pressing problem. In the first
nine months of 2023, inflation averaged 6.6% — far above the upper end of the target range of 2-4% set by the BSP. If left unchecked, inflation could undermine
growth. Not only does this increase costs to organizations, but it also sets in motion second-round effects as workers start demanding higher wages,
consequently increasing business costs and discouraging investments.
</item 4>

<item 5>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Throughout September, in Tokyo, New York, Hong Kong and London, the world’s biggest investment banks hosted a series of conferences for the world’s biggest
investors, all gravitating around the same theme: is Japan really back?

And, according to both the attendees and hosts of those events — which were run by JPMorgan, Goldman Sachs, Bank of America, Mizuho, BNP Paribas, and Jefferies
— the answer is now clearer than it has been in a very long time. Japan is back, but global investors may take a little more persuasion that it is truly back
for the duration.

So, this week, prime minister Fumio Kishida will make his strongest ever push for a bet on Asia’s biggest advanced economy — as BlackRock founder Larry Fink,
and other heads of global funds, gather in Tokyo for a series of events to attract investment.
</item 5>

<item 6>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
In Fiscal Year 2022, this amounted to P74 billion.

Moreover, under the Universal Health Care Law, PhilHealth receives 50% of the government share from the income of the Philippine Amusement Gaming Corp. (PAGCOR)
as well as 40% of the Charity Fund of the Philippine Charity Sweepstakes Office (PCSO). In 2022, this amounted to P188 billion.

In like manner, in addition to the school vouchers funded under the Expanded Government Assistance to Students and Teachers in Private Education Act, we suggest
that PhilEd be allocated a share of the taxes imposed on our extractive industries. This is based on the argument that the depletion of our natural resources
can only be justified if the taxes collected will enhance our human resources.

Like PhilHealth, PhilEd could also be mandated to receive a percentage from the collection of documentary stamps which have recently been raised.

Finally, PhilEd can be given access to the Special Education Funds of the local school boards of the Local Government.
</item 6>

<item 7>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
It will also promote a green and blue economy coupled with improved governance to guarantee long-term climate and disaster resilience.

BUILDING RESILIENCE THROUGH SUSTAINABLE DEVELOPMENT

Coming off from the pandemic and with the current global economic climate, it is opportune for the government to proactively drive for actionable policies and
programs that focus on building the resiliency of the economy through sustainable development with greater emphasis given to addressing climate change.

Navigating external and domestic economic headwinds will not be an easy feat for the administration, but the private sector will be an important catalyst for
sustainable development. With the private sector sharing industry knowledge, resources and potentially even leading certain socio-economic programs and projects
of the National Government, it is to be hoped that AmBisyon Natin 2040 of long and healthy lives for Filipinos that are strongly rooted, comfortable, and secure
will be achieved.

This article is for general information only and is not a substitute for professional advice where the facts and circumstances warrant. The views and opinions
expressed above are those of the author and do not necessarily represent the views of SGV & Co.
</item 7>

<item 8>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
“Deployment of capital to China has been a bit more risky in terms of eventual outcome of investment decisions, given that lack of clarity.” But interest in
Japan has not only come about because of the push away from China, he adds.

Japanese companies are increasingly aligning themselves with the interests of shareholders in a way that once seemed unlikely. Part of that, he says, is
generational. Companies that replace their chief executive on five or seven year cycles now have leaders that reached their positions after the establishment of
Japan’s stewardship code (in 2014) and corporate governance code (2015). Many have a very different view of the role of a chief executive than their
predecessors did.

“I think that Japan, in terms of the unifying message that companies and the Tokyo Stock Exchange are sending out, now gives a lot more confidence about where
they are deploying capital,” says Neureiter. “So people looking at the region are finding Japan incrementally more attractive.
</item 8>

<item 9>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
We recently presented a paper at the 5th Paderanga-Varela Memorial Lecture entitled “Alternate Service Delivery Systems: Health and Education.” In that lecture,
we looked at the Department of Health (DoH) and the Department of Education (DepEd) as service delivery systems, delivering healthcare and education
respectively.

We concluded that the service delivery system of the DoH is far superior to the DepEd for two main reasons. The first is that the DoH has devolved its hospitals
to the local government units while the DepEd has not. Secondly, the DoH has spun off its health financing into the Philippine Health Insurance Corp.
(PhilHealth) while the DepEd has not. In this article we argue for the creation of the Philippine Education Development Fund (PhilEd).

When Isidro Consunji, President of DMCI Holdings, Inc. was asked by his officers what the role of his Chief Financial Officer (CFO) would be, he gave a
three-word answer, “Fund my dreams.”
</item 9>

<item 10>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Inflation is partly driven by supply-side issues and the government is allocating more resources to the agriculture sector to boost production. Mr. Marcos also
mentioned that the National Government had distributed 28,000 new tools and machinery to farmers. An additional 600 km of farm-to-market roads were laid down to
support the 14 million hectares of farmland, enhancing farmer access to markets. In addition, he signed Executive Order No. 28 in May, forming the Inter-Agency
Committee on Inflation and Market Outlook, which is tasked to keep inflation within government targets and boost the economy.

ENHANCING CLIMATE RESILIENCE

While price stability and fiscal sustainability are important macroeconomic issues, environmental sustainability is increasingly gaining importance. Climate
change uncertainties and challenges need to be managed and both the Philippine government and the private sector are hard-pressed to deliver their commitments
to addressing them. Despite climate change risks, the Philippines has the opportunity to position itself as a prime destination of foreign investments against
climate change or environmental, social, and governance (ESG) investments.
</item 10>

<item 11>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
Why then, do we have a deficit budget for 2024?

The national budget for 2024 is at P5.768 trillion, 9.5% higher than the current (2023) budget of P5.268 trillion. Observers note that this first budget fully
prepared by the current administration accounts for 21.7% of gross domestic product (GDP). The 2024 expenditure program is higher by P499.6 billion or a
year-on-year growth rate of 9.5% as compared to the 2023 level of P5.268 trillion. The P5.768-trillion budget, to be supported mainly by P3.5-trillion tax
revenues (based on 2023 tax revenue projections), would mean a budget deficit of some P2.268 trillion, to be funded by additional borrowings. This could
increase government debt from the current P14.24 trillion (as of July) to P16+ trillion by the end of 2024.
</item 11>

<item 12>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Noel P. Rabaja the strategy and transactions (SaT) service leader of SGV & Co.
</item 12>

<item 13>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).

Whom are we to believe, but inflation — which independently says it as it is.

“While inflation in the Philippines has been on a downward trend since January, the country needs to achieve a 3.7% average inflation rate for the next five
months for the country to hit the BSP’s forecast of 5.5% inflation for 2023, Assistant National Statistician Rachel Lacsa said (Ibid.). Some economists say that
deficit budgets and the remedies — monetary (borrowings increasing money supply) and fiscal (like raising taxes) increase inflation, and stubborn inflation
further entrenches structural deficits in the system. It is a vicious cycle.”

National budgets might not be innocent of political agendas (call them plans and programs) that can divert money from the proper socio-economic goals of
equilibrium as a balanced budget might work towards.
</item 13>

<item 14>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
PhilEd will be the “bastonero” or project manager, managing the project as well as mobilizing government support and assistance.

FAPE or the Fund for Assistance to Private Education was a part of the Special Fund for Education set aside by the American government in 1963 from the surplus
funds authorized by the War Damage Act of 1962 and organized under Executive Order 156 issued by President Ferdinand Marcos on Nov. 5, 1968.

From the initial seed money of $6,154,000 or P215 million (based on the 1968 peso to dollar exchange rate), the FAPE fund in 55 years had barely grown to P277
million. In contrast the trustees managing the fund have expanded their services, basically as service provider to the DepEd. As a consequence, it has been spun
off as a separate organization, the Private Education Assistance Committee (PEAC).
</item 14>

<item 15>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
For the energy sector in particular, amendments to the PSA are envisioned to help raise the capital needed to speed up the energy transition of the country to
renewables.

In addition, the Regional Comprehensive Economic Partnership (RCEP), a trading bloc that encompasses the ten members of the Association of Southeast Asian
Nations (ASEAN) and the ASEAN Plus One Free Trade Agreement (FTA) partners Japan, China, South Korea, Australia, and New Zealand, is already in force for the
Philippines, helping ease market access through trade and investment rules and supporting global and regional supply chains. The Philippines can potentially
position itself as a regional manufacturing hub if the right incentives and policy measures are put in place to encourage local and foreign investors to
participate. Together, trade and investment are expected to play an important role in attaining economic transformation, the second goal of the PDP.

HUMAN CAPITAL AND INFRASTRUCTURE

The ability of people to take advantage of market opportunities arising from investment rests on human capital.
</item 15>

<item 16>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
“After extensive deliberations and negotiations, the House of Representatives on Wednesday (Sept. 27) approved on third and final reading the P5.768-trillion
national budget for next year, just before Congress adjourns for a month-long break,” the Press and Public Affairs Bureau of Congress announced on its website,
congress.gov.ph. The 310-strong legislative chamber voted 296 against three, with zero abstentions, to approve House Bill (HB) No. 8980 or the 2024 national
budget on third and final reading. Almost unanimously.

House Speaker Ferdinand Martin G. Romualdez noted that the budgeting process was marked by rigorous discussions, particularly concerning confidential and
intelligence funds, which the House meticulously scrutinized to ensure accountability and responsible utilization. He lauded “the timely passage of the 2024
General Appropriations Bill (GAB),” stating that it not only “demonstrated the commitment of the House to upholding its constitutional role but also underscored
its dedication to serving the Filipino people with transparency and accountability.”
</item 16>

<item 17>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Public policies attempt to fulfill one or more of the following objectives: efficiency, equity, and stability. Efficiency and stability are necessary conditions
for sustainable growth while equity helps make development inclusive.

This is the second article in our series following the 2nd SGV Tax Symposium, which focused on how a sustainable and effective tax ecosystem can advance the
sustainability agenda for both the public and private sectors.

In this article, we will discuss the Philippine strategy for sustainable development.

GROWTH THROUGH TRADE AND INVESTMENT

The Philippine Development Plan (PDP) 2023-2028 aims for a deep economic and social transformation to reinvigorate job creation and accelerate poverty reduction
by steering the economy back to a high-growth path. In this regard, two main performance indicators are identified and will be monitored by the National
Government.

The first is for the country to graduate into upper middle-income class status within the term of the current administration.
</item 17>

<item 18>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Congress enacted Republic Act 7875, known as “The National Health Insurance Act of 1995,” which created PhilHealth. We are hoping some members of Congress would
file a Universal Education Financing Law which would create PhilEd.

Dr. Victor S. Limlingan is a retired professor of AIM and a fellow of the Foundation for Economic Freedom. He is presently chairman of Cristina Research
Foundation, a public policy adviser and Regina Capital Development Corp., a member of the Philippine Stock Exchange.
</item 18>

<item 19>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
The Department of Budget and Management (DBM) defines the government budget as “the financial plan of the government…which shows what its resources are, and how
they will be generated and used over the fiscal period. It refers to the income, expenditures and sources of borrowings of the National Government (NG) that are
used to achieve national objectives, strategies and programs.”

Section 22, Article VII of the Constitution states that: “The President shall submit to the Congress within 30 days from the opening of every regular session,
as the basis of the general appropriation bill, a budget of expenditures and sources of financing including receipts from existing and proposed revenue
measures.”

The National Government (NG) budget is the totality of the budgets of all government entities, namely the National Government, the non-financial
government-owned and -controlled corporations (which usually includes only the 14 major GOCCs), government financial institutions (GFIs), local government units
(LGUs), the social security institutions (the Government Service Insurance System or GSIS, the Social Security System or SSS, the military pensions, PhilHealth,
etc.
</item 19>

<item 20>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The government calls for embedding resilience, sustainability, and nature-based solutions in infrastructure planning and investment to enhance climate
resilience. Likewise, investments in renewable energy are expected not only to enhance energy security but also reduce greenhouse gas (GHG) emissions.

From an economic vantage point, GHG emissions are economic “bads” and are a cause of inefficiency as they get over-produced. Their effects, however, go beyond
national borders, and while the Philippines contributes minimally to global GHG emissions, the World Risk Index 2022 report identifies the Philippines as the
most disaster-prone country in the world.

The government envisions that by 2028: (1) Climate and disaster risk resilience of communities and institutions will increase (2) Ecosystem resilience will be
enhanced, and (3) A low-carbon economy transition is enabled.

To ensure that these are realized, the government is set to safeguard cross-sectoral convergence and implement a comprehensive risk management approach to
address the adverse consequences posed by climate change.
</item 20>

<item 21>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
How can this larger deficit be financed?

"It’s reasonable to argue that besides the hope of financing part of the deficit through aid from the US (President Biden approved $8 billion military aid this
week, H. S.), and that will happen, it’s permissible for a country at war to raise its fiscal deficit by what is seen as the right amount. This is in order to
ensure that the economy functions properly, to carry out rehabilitation actions, and to prepare for recovery and a return to growth in the near future."

How high should the deficit go?

"I mean a deficit of 3% of GDP, or 5%, or 8% - there are no sacred numbers. As was said during the Covid pandemic, the state has to do whatever it takes to
achieve these goals. Nevertheless, it’s important to make sure that the growth in expenditure and in the deficit is temporary, until economic activity gets back
to normal.
</item 21>

<item 22>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
For this, the economy must grow its per capita income above the threshold set by the World Bank, which means a gross national income (GNI) per capita of at
least $4,466. In 2022, the Philippines achieved a GNI per capita of $3,950.

The second is to lower the poverty level from 18.1% in 2021 to single digits by 2028 — the end of the term of the current administration. Both indicators
require high growth rates. For the next year through 2028, the government pencils the growth rate between 6.5% and 8%.

Growth is expected to be investment-led with the implementation of structural changes such as the Corporate Recovery and Tax Incentives for Enterprises (CREATE)
law, which lowered the corporate income tax rate, and the amendments to the Public Service Act (PSA), Foreign Investment Act (FIA), and Retail Trade
Liberalization Act (RTLA), which further liberalized the economy. The new legislation is expected to attract more local and foreign investment, especially in
the liberalized sectors.
</item 22>

<item 23>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
The surplus budget policy is important to encourage economic growth,” the DBM said.

The exhortations of the national budget managers must be quoted word for word to educate the non-economists and questioning common people about how a surplus
budget is necessary versus a deficit budget:

“The less the government borrow from the public, the lesser the pressure on interest and inflation rates and the more funds are made available in the financial
market. Such funds may be used by businessmen to build factories, hire workers, buy equipment and open more employment opportunities. By keeping more funds in
the hands of the private sector rather than competing for credit, the government helps make financing available for families who want to own homes, buy cars, or
support their children’s education. The government also needs to generate a budget surplus to repay the huge debt it has accumulated over the years. The
reduction of the national budget debt will correspondingly lessen government’s requirements for interest and principal payments. This becomes important
particularly during periods of rising interest rates and unstable exchange rates” (dbm.gov.ph).
</item 23>

<item 24>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
“I would urge you to evaluate what we are doing in my country, look at the underlying strength of our economy and our plans for the future and then invest in
Japan,” Kishida said in a recent speech at the Economic Club of New York.

On the face of it, the Tokyo conference guests have been told by various experts that the stars do appear aligned in Japan’s favour, both on its own merits and
in comparison with what is happening around it. But, because so many global funds have diverted expertise away from Japan in recent years, they are concerned
that they need to know more, as a matter of urgency.

“There is heightened interest in Japan after what has been, for many, a long period of not being focused on that country,” says Paul Hitchens, the head of the
Japan research department at Jefferies, which hosted a major investor conference in Hong Kong.
</item 24>

<item 25>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
During the Congressional budget hearing, Finance Secretary Benjamin Diokno optimistically provided updates on the government’s fiscal and revenue collection
performances and presented sources of funding for the national budget. Tax revenue is projected to increase from P3.5 trillion in 2023 to P6.5 trillion in 2028,
while non-tax revenues are expected to decline from P191 billion in 2023 to P184 billion in 2028, Diokno told the Philippine News Agency (pna.gov.ph, Aug. 10).

Diokno said the Finance department aims to gradually reduce net financing or new debt from P1.42 trillion in 2022 to P1.22 trillion in 2024, which will help
bring the country’s debt-to-GDP ratio to almost below 60% by 2025. And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63%
and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr.
</item 25>

<item 26>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Japanese prime minister Fumio Kishida will appear this week at a series of events aimed at attracting foreign investment © AFP via Getty Images

There are distinctive catalysts for that, he adds — noting a recent push by the Tokyo Stock Exchange for companies to improve their capital efficiency, and the
high profile decision of Warren Buffett to invest heavily in five of Japan’s largest trading houses.

Hitchens says: “Is there some healthy scepticism? Yes. However, global investment firms know that they need to invest and build their offerings in Japan to meet
the growing appetite, and that they need to pay attention to an equity market at a 33-year high.”

Topix, Japan’s main stock index, has risen more than 25 per cent this year, outperforming almost all the world’s developed market indices and, as some
conference attendees admit, creating a “fear of missing out” on a rally that may have significantly further to go.
</item 26>

<item 27>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Large global funds are not the only ones to have had their interest in Japan piqued, says Keith Truelove, head of global markets at UBS Securities Japan.

“There is still a lot of dry powder on the sidelines,” he observes. “There is a lot of interest in Japan from family offices and from high net worth individual
investors, who value wealth preservation more highly than institutional investors — it’s really taking off now.”

“They took their time to decide if it was safe to invest in Japan again, because they need to get confidence in the idea that the Japan story will be more
sustainable this time,” Truelove adds. He notes that Japan has offered investors a number of false-dawns over the years, and the bar for deciding that ‘this
time is different’ remains high.

One of the most significant changes, say asset managers, is the fundamental shift in the way global funds now look at China.
</item 27>

<item 28>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
But traditional politicians (called “trapo” in the Philippines) might have it in their hearts to think about the country and the welfare of future generations.

How many generations will pay for the monstrous debt incurred for the dubious objectives and vanities of some today?

Amelia H. C. Ylagan is a doctor of Business Administration from the University of the Philippines.

ahcylagan@yahoo.com
</item 28>

<item 29>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
The geopolitics of US-China tensions, chip wars, decoupling and “friendshoring” seem, in particular, to be working significantly in Japan’s favour. Asia-focused
global investment is rotating out of China and into Tokyo stocks, real estate and other assets, just as large parts of corporate Japan are engaging more
actively with shareholders than at any time in the past.

China, after years of stunning growth and investor attention, is very rapidly losing that sheen. Its economy is growing more slowly and the expected
post-pandemic rebound is proving elusive. On top of that, Beijing’s policymaking — with its sudden crackdowns on entire industries, like video games and private
tutoring, has generated a series of nasty surprises — and the unshakeable fear that others could occur at any time.

“You are seeing more interest in Japan from investors, who were primarily invested in China, as China has slowed and some of its economic policies have been
confusing and opaque,” says Kirk Neureiter, president of Fidelity Management & Research Japan.
</item 29>

<item 30>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
), the Oil Price Stabilization Fund, the Bangko Sentral ng Pilipinas, and the Central Bank-Board of Liquidators. It is allocated for the implementation of
various government programs and projects, the operation of government offices, payment of salaries of government employees, and payment of public debts. These
expenditures are classified by expense class, sector and implementing unit of government (dbm.gov.ph).

The sources of appropriations of the annual budget are: 1.) new general appropriations legislated by Congress for every budget year under the General
Appropriations Act (GAA); and 2.) existing appropriations previously authorized by Congress. Under the Constitution, Article VI, Section 29, no money can be
withdrawn from the Treasury except in pursuance of an appropriation made by law.

The DBM calls a budget balanced when revenues match expenditures or disbursements. When expenditures exceed revenues, the government incurs a deficit which may
result in the following situations: a.)
</item 30>
=== END ITEM id=4e09a60f573969a8 ===
