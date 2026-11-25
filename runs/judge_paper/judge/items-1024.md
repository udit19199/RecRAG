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

=== ITEM id=4dd04af83e70f994 ===
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
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Oz. But his TikTok presence consisted of reformatted TV ads, rants attacking his opponent, now-Senator Fetterman, and unrelatable videos that inadvertently
portrayed him as wealthy and out of touch.

There have been Republican outliers who managed to build significant followings, but so far, none have managed to turn their social engagement into votes the
way Trump’s Twitter account did in 2016. Pennsylvania Sen. Doug Mastriano shunned traditional media during the race, instead relying on Facebook Live to build a
far-right grassroots network that secured him the Republican nomination in the state’s most recent gubernatorial race. He still lost to Josh Shapiro, whose
campaign focused on reaching young voters through collaborations with Gen Z for Change, visiting college campuses and posting updates on BeReal.

There’s clearly an audience for right-wing ideology online, as conservative influencers continue to build massive platforms, largely by provoking outrage.
</item 1>

<item 2>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
A number of works outline the contours and dynamics of the current media ecology and what it does — Siva Vaidhyanathan’s Antisocial Media, Safiya U. Noble’s
Algorithms of Oppression, Geert Lovink’s Sad by Design, Shoshana Zuboff’s Surveillance Capitalism, Richard Seymour’s The Twittering Machine, Tim Hwang’s
Subprime Attention Crisis, Tressie McMillan Cottom’s writing on how to understand the social relations of Internet technologies through racial capitalism, and
many more. At the same time, there’s reasonable counter-discussion about how many of our problems can really be laid at the feet of social media. The research
on the impacts of social media on political dysfunction, mental health, and society writ large does not paint a neat portrait.
</item 2>

<item 3>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
Now they can do the same for social networks. Image: Andrew Parker

This is the same opportunity in front of the social media landscape right now: a rare chance to unbundle the internet, to pull apart an existing system and
rebuild it, piece by piece, in vastly better ways. If we do this correctly — if the next phase of how we congregate and communicate online is built for humans
and not advertisers — there won’t be a new titanic company to rival Meta or a platform with eye-poppingly huge numbers like Facebook. What we’ll get instead is
something much bigger: an entirely new infrastructure for our online lives that no company or platform controls.

The reason this suddenly feels possible is the emergence of the fediverse. In a sentence, the fediverse is an interconnected set of apps that can all read and
write the same content.
</item 3>

<item 4>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
On Equity this week, the crew talked about the SBF trial; deals from VR firms Rainforest, At One Ventures, Section 32 and Greylock, where venture funding has
declined; and how Fearless Fund, a firm founded to invest in women of color, is being barred from awarding grants to Black women founders.

Meanwhile, Found featured Esther Rodriguez-Villegas from Acurable, a medical device company that makes patient-friendly wearable devices to diagnose and manage
respiratory conditions at home. As a career-long academic, Rodriguez-Villegas talks about how she never intended to be a founder until she learned about how the
currently available medical devices make it extremely difficult to detect and treat diseases like sleep apnea and epilepsy.

And over on Chain Reaction, Jacquelyn did a crossover episode with Alex about the SBF trial.
</item 4>

<item 5>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
With Amo, the founder of Zenly wants to make social apps social again The startup raised $20 million and is launching ID today

In 2018, I wrote a TechCrunch article stating that 2018 was “the year social networks were no longer social.” Reflecting on that article, I’m not sure that 2018
was the turning point. But the premise of the article still holds up well.

At some point, social networks were no longer about connecting with your closest friends, keeping up with long-distance family members and feeling a special
connection with people you love.

TikTok, Instagram, Snapchat, Facebook, YouTube and X (formerly Twitter) are all slowly evolving to become the same infinite scrollable feed of
algorithm-optimized short videos from top performers.

And it turns out that I’m not the only one who has noticed that social networks have been slowly drifting away from their original purpose.
</item 5>

<item 6>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
Pure fediverse, baby.

I’m convinced we’ll be better off with a hundred different apps for Snapchat or Instagram or X instead of just one

I’m convinced we’ll be better off with a hundred different apps for Snapchat or Instagram or X instead of just one, a dozen companies competing to build the
best moderation tools, and an app store filled with different ways for me to follow and be followed by other people on the internet. It doesn’t make sense that
we have a dozen usernames, a dozen profiles, a dozen sets of fans and friends. All that stuff should belong to me, and I should be able to access it and
interact with it anywhere and everywhere.

The infrastructure underlying all of this is typically ActivityPub, a decade-old protocol overseen by the World Wide Web Consortium (also known as the group
more or less in charge of how the internet works). There are other similar protocols as well, like Bluesky’s AT Protocol and Nostr and Farcaster.
</item 6>

<item 7>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 7>

<item 8>
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
</item 8>

<item 9>
How the conspiracy-fueled Epoch Times went mainstream and made millions (Hacker News, 2023-10-16T03:41:24+00:00)
Through the early aughts, The Epoch Times grew from an online effort to a weekly physical newspaper, with a home base in New York and a TV production company,
New Tang Dynasty Television. It raised money from followers and was staffed by unpaid volunteers. It ran aggregated articles on international issues from Voice
of America next to Thanksgiving Day explainers, dispatches from Falun Gong parades, and exposés on atrocities alleged to have been committed by the Chinese
Communist Party.

By 2019, it had gone mostly digital and was spending millions of dollars on creating a network of Facebook pages and groups and running aggressive pro-Trump ad
campaigns. The move toward explicit support of Republicans, despite Li’s teachings to stay away from U.S. politics, was foreshadowed by Li’s comments at a Falun
Gong conference a year before.

Li said that Falun Gong’s media ought to put a “constructive” spin on the news, to advance the group’s aims.
</item 9>

<item 10>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.
</item 10>

<item 11>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
SBF allegedly tried to buy off Trump: The TC team’s been trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced entrepreneur
accused of orchestrating the collapse of cryptocurrency exchange FTX. But fascinating details about SBF’s political dealings are emerging from a book by Michael
Lewis, “Going Infinite,” that debuted on the first day of the trial, like SBF’s attempt to buy off Trump to get him to not run again for president.

Gmail fights back against spammers: Google this week announced a series of significant changes to how it handles email from bulk senders in an effort to cut
down on spam and other unwanted emails. The company says that, starting next year, bulk senders will need to authenticate their emails, offer an easy way to
unsubscribe and stay under a reported spam threshold.

TikTok tests an ad-free tier: TikTok is testing an ad-free subscription tier for some users.
</item 11>

<item 12>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
“I feel like the majority of my life has been in front of a screen,” Turner “Tfue” Tenney reflected amid tears. A fateful kill against Tyler “Ninja” Blevins
back in 2019 had catapulted Tenney to fame, and his audience went on to outgrow that of Twitch’s most mainstream star. Yet in 2022, Fortnite’s golden boy —
whose name is synonymous with the battle royale game’s most recognizable character, Jonesy — called it quits. For many, Tenney’s retirement video was a sign of
yet another Twitch star grappling with burnout: sad, yet familiar. In essence, though, Tenney walking away from it all cemented the end of an era for the games
industry.

Polygon is celebrating TwitchCon 2023 with a package of stories covering streaming trends and news from the show. Check it out!

Tenney was the real deal, a perfect nexus of undeniable charisma and tenacious talent on display at least eight hours a day for years.
</item 12>

<item 13>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
That’s one takeaway, at least, from the recent shutdown of Pebble (formerly T2), a Twitter clone that put trust and safety at the forefront of its development
process. The app never gained more than 20,000 users, and the founders now believe that its approach to moderation may have been correct, but it wasn’t a growth
driver. In other words, people may say they want to feel safer when participating online, but clearly, that isn’t their only need.

Mozilla’s instance, still in private testing, has a small team of moderators that will grow when the instance launches to the public in the coming months.

“It’s not just a commitment, it’s beyond foundational for us — that is the thing that is going to be true about this instance,” O’Hara says of Mozilla’s plans
to heavily moderate its space. Though Mozilla will also rely on technology to do some of the work, it wants to make sure the human moderators, “feel protected,
supported and respected for their work,” she adds.
</item 13>

<item 14>
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
</item 14>

<item 15>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 15>

<item 16>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The grip Twitter, now called X, has on the market may be stronger than some believed. Unfortunately, that’s led to the first casualty among Twitter
alternatives, as the startup Pebble (formerly T2), is shutting down. The would-be X rival had grown a small but engaged community on its microblogging service
that aimed to dupe Twitter’s features, from its verification systems to functionality like DMs. But the company says they’ve run out of time to make Pebble
happen — the app maxed out at 3,000 daily active users, out of 20,000 registered users. That daily user figure fell to 1,000 daily users following its
rebranding from T2.

In part, the sizable competition coming from other Twitter rivals was to blame, says co-founder and CEO Gabor Cselle, who had exits to Google and Twitter prior
to starting Pebble.

“I think the competitive landscape evolved faster than we had thought,” he explains.
</item 16>

<item 17>
People are turning to Snap Map for firsthand perspectives from Gaza (TechCrunch, 2023-11-03T17:24:18+00:00)
Snap Map has served as an alternative source of eyewitness accounts during times of social unrest, as young people increasingly perceive news media reporting as
“biased.” Trust in legacy media is waning in younger generations — a 2022 Pew Research Center survey found that American adults under 30 trust information from
social media nearly as much as they trust information from national news outlets. A report conducted by the Knight Foundation earlier this year concluded that
perceptions of political bias in news have increased, and young people hold more negative perceptions of news media than their predecessors. Younger people also
consume the most online news, but report the lowest trust in national news outlets.

Although Snapchat isn’t immune to misinformation, it’s worth noting that spreading fake content via Our Story isn’t as easy as reposting a video on X or TikTok.
Users can upload content from their camera roll to their own Stories, but it won’t be eligible for Our Stories. Snap encourages users to “capture what’s
happening” around them to increase their chances of making it to Our Stories.
</item 17>

<item 18>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Third-party Reddit app Narwhal says it hopes to survive Reddit’s app purge with a subscription plan.

Apropos “getting together,” Amanda reports that VidCon is still kickin’. For the first time, the conference hosted an industry leadership summit, where creator
economy experts and creators could hash out their grievances with the state of the business and share ideas to make the job of a creator more sustainable. That
makes sense, unlike creators raising venture capital: It’s so eye-wateringly hard to make money as a creator, I’d love to see the pitch that convinces a VC to
cut a check to a creator, with a realistic expectation of a venture-scale return.

Maybe they were hiding behind the sofa?: Sarah reports that Mastodon actually has 407K+ more monthly users than it thought — and it seems like Twitter has a lot
more traffic than Musk said. Still, the peak now is about the same as it was a decade ago, and it’s unclear what the social platform can do to encourage more
growth.
</item 18>

<item 19>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Critics on both the right and the left described the launch as a disaster.

Conservative organizations have proven especially out of their depth when facing the wrath of extremely online social media users, who mobilize trolling for
social justice. In 2020, TikTok users reserved hundreds of tickets for a Trump rally and never showed up, leaving the president to face swaths of empty seats.
When a Texas anti-abortion group opened an anonymous tip form to enforce the state’s ban on abortion after six weeks, Gen Z activists flooded the site with
Shrek porn. This year, TikTok and Twitter users shut down the Missouri attorney general’s tip form for reporting clinics that provide gender-affirming care.
Within days of launching, the site was spammed with fanfiction, satirical anecdotes about kids getting “transed” and fanfiction. The attorney general’s press
secretary blamed the site’s downfall on activists “hacking the system.”
</item 19>

<item 20>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
Initially, Flipboard is testing the integration with select accounts, including publishers like Semafor, Pitchfork, Fast Company, Medium, LGBTQ Nation, Refinery
29, Digiday, Polygon, SPIN, Kotaku, Frommer’s, The Verge, Smithsonian Magazine, Refinery 29, The Root, ScienceAlert, AFAR Media and others. While many are
focused on news, there are also some nonprofits like The News Literacy Project and education-focused news site The 74, in this debut list.

“As we said in the earlier part of this year, we will be embracing ActivityPub into Flipboard and effectively reworking our whole backend to that,” explained
Flipboard CEO Mike McCue, in a conversation with TechCrunch about the coming changes. He says the company had first integrated with Mastodon at the API level,
so users could log into their Mastodon accounts, see those posts and interact with others in the fediverse from Flipboard. “But you had to have an account on
all those platforms,” McCue noted.
</item 20>

<item 21>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
Threads, Instagram’s supposed “Twitter killer,” got 100 million sign-ups within five days; the decentralized platform Mastodon has 1.8 million monthly active
users; and the smaller upstart Bluesky hit 1 million users in September. Though still in beta, Spill’s community of 200,000 users may seem small in comparison,
but Spill’s rise has shown that quality matters just as much as quantity — that building a strong core community is just as valuable as high-speed growth, even
if the traditional venture model may not always incentivize thoughtful product development.

“If you have a great experience, you’re going to grow,” Terrell said. “Especially in a world where you have so many networks — there are all these choices that
people conceivably have — you’re gonna go to the one where you have the greatest experience and want to invest your time.”
</item 21>

<item 22>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
As it learns from the publishers and other content creators what they want and need, it then intends to use that understanding to build features and drive
efforts that can help them reach their audiences.

So far, what Mozilla has learned is that publishers today feel somewhat resigned and pessimistic about social networks. Twitter isn’t driving much traffic and,
even though Instagram Threads appears interesting, Meta specifically said it wasn’t going to prioritize news on that platform. What Mozilla wants to accomplish,
then, is to help reconfigure the Mastodon onboarding process so that when someone — including a publisher or creator — joins its instance (or the fediverse in
general) they’re able to build their audience with more ease.

“The baseline is just better onboarding, better connecting with accounts and communities,” explains O’Hara. “You might be interested in spaces where you can
have the kinds of conversations that you want to be able to have; trust and safety practices that make you feel safe in order to participate. And I think also
then access to content that spurs those conversations and seeds those conversations.
</item 22>

<item 23>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The founders don’t regret their experience building Pebble, even though it didn’t turn out as they hoped, they said.

Says Cselle, “One thing that I learned is that there’s absolutely an audience that wants to see a new kind of Twitter-like platform built and will ask for the
features that Twitter has.” He suspects the team might stay together to work on something new after Pebble wraps, taking the learnings from Pebble with them.
That includes better communicating to users what great content should look like on a platform, and how they can win.

“We’re at an inflection point in social media,” says Oh. “We had one hypothesis with T2 and Pebble. It obviously didn’t pan out the way that we wanted it to.
But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
</item 23>

<item 24>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 24>

<item 25>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
If anything, Ohanian credits Twitter (now called X) with coming up with the best system for moderation so far.

“Community Notes, I personally think is a good — it’s the best measure I’ve seen yet. And frankly, I wish we had thought of it first,” he said.

Still, he called out the notion that a platform, like X, can be a town square.

“The public square [analogy]…falls flat. Because if that then implies that it’s a public square, which means like, what? a town or a city? If you’re the person
in charge of it, you’re the mayor. But there’s no democratic election,” Ohanian pointed out. “You’re a CEO. So then you’re like the king? If you want to lean
into the town square thing, fine. Just own it and say, yeah, it’s a town square and I am the infinite dictator, tyrant of it.”
</item 25>

<item 26>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The software development startup previously raised $35 million in a Series B funding in 2022.

Bill

Will cut around 15% of its workforce, the company announced December 5, and close its Sydney office to increase profitability.

CourseHero

Announced December 5 it eliminated 23 employees across engineering, operations and development.

Pivo

Shut down operations December 5. The Nigerian fintech startup previously raised $2 million in a seed round.

Twilio

Announced hundreds of job cuts December 4, amounting to 5% of the company’s workforce.

Spotify

Is eliminating about 1,500 jobs, the company announced December 4. This marks Spotify’s third round of layoffs this year.

Bending Spoons

Has laid off the entire staff of Filmic, the team behind popular video and photo-editing apps that Bending Spoons acquired in July.

November 2023

Zepz

Cut 30 roles November 30 after the fintech unicorn eliminated 26% of its workforce earlier this year.
</item 26>

<item 27>
X plans to use payment, phone and ID verification to stop bots, in addition to the new $1/yr fee (TechCrunch, 2023-10-18T17:33:48+00:00)
A report this week from market intelligence firm Similarweb, for example, found that X’s web traffic and monthly active users on mobile had declined
post-acquisition.

Users today have far more options for microblogging and public discourse, thanks to products from Twitter-like startups and tech companies such as Spill,
Bluesky, Pebble, Countersocial, Spoutible, Hive and the open source platform Mastodon, as well as efforts from tech giants, like Meta’s new app Threads. Few of
the other social apps currently charge a fee, unless it’s for premium features, like some of the third-party Mastodon apps do.

In addition to presenting friction at sign-up, other critics have argued a fee could exclude genuine users and increase the digital divide.
</item 27>

<item 28>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
It also sees a role for its read-it-later app Pocket in this experience, as the app’s primary use case is to store links to articles and other content that
people have signaled they want to read.

“We are planning to open the beta in phases because we want to ensure that we have a content moderation team and other tools that can be scaled relative to the
size of the user base. We’re not in a rush as we think it’s important to be deliberate about launching social media tools,” O’Hara says.

“We have a lot of core competency that we brought over from our Pocket experience and other recommendation products that we put together to at least get that
engine revving,” notes O’Hara.

Initially, the company is running a Mastodon account, “@Pocket_Recommends@mozilla.social,” where it actively posts recommendations of stories to read. It’s the
same sort of content you might find in the Pocket email newsletter, but now distributed on the fediverse.
</item 28>

<item 29>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Tropic

Eliminated 26 roles, CEO David Campbell confirmed in a LinkedIn post October 21.

Convoy

Is halting operations due to a “massive freight recession,” CEO Dan Lewis announced in a memo October 19, a week after laying off 20% of its workforce.

LegalZoom

Plans to lay off more than 100 workers in its sales division, the company disclosed in a Worker Adjustment and Retraining Notification Act letter.

Plume

Laid off more than two dozen workers, TechCrunch learned October 18, about a year after the trans healthcare startup raised a $24 million Series B.

Google

Eliminated at least 40 roles in its news division, Google confirmed October 18.

Expedia Group

Cut around 100 jobs, Skift reports, in its second round of layoffs in recent months. The cuts reportedly impacted employees across data and AI.

Stack Overflow

Laid off 28% of its staff, more than likely impacting 100 people, the Prosus-owned company announced October 16.
</item 29>

<item 30>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
The attorney general’s press secretary blamed the site’s downfall on activists “hacking the system.”

The presidential election is more than a year away, but it may be too late for Ramaswamy to make any headway on TikTok, where the platform’s young users still
don’t take him seriously. He recently posted a video about getting ready for the next Republican debate, and was hit with yet another wall of edging comments.

Even if candidates like Ramaswamy did everything right — like having a platform that didn’t alienate young voters and working with creators with more favorable
reputations — they’d still represent a party that many Gen Z and millennial voters aren’t aligned with.

“It’s really hard to be effective with a generation when a lot of your policy is attacking them,” Henry said. “If all of your policy is highly unfavorable for
that generation, that’s a hard sell in itself, even if you’re an effective communicator with all the strategy in the world, to sell someone something they don’t
want.”
</item 30>

<item 31>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
Cowards.

A swan song from the bird sanctuary: X has marked its value at $19 billion, according to internal documents, marking a 56% decrease in value since Elon Musk’s
purchase last year. Bad for shareholders and Musk, but potentially good for executives and employees, Amanda explains.
</item 31>

<item 32>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think the competitive landscape evolved faster than we had thought,” he explains. “I didn’t think that quite as many people — established organizations and
newcomers — would try to do the same thing that we were doing and in very similar ways.”

Today, the market for Twitter alternatives is a crowded one, to say the least. There are numerous services for consumers to try, including the open source-based
platform Mastodon, a soon-to-be decentralized system from Bluesky, plus smaller startups like Spill, Spoutible and Post, as well as a new app from Meta called
Instagram Threads.

Then, of course, there’s the X factor to consider. The default platform for short posts retains some pull, despite — or perhaps because of — new owner Elon
Musk’s antics. Though the company may not have yet succeeded from a business and financial perspective, it’s been hard for others to duplicate its function as a
breaking news platform and place for spirited debates.
</item 32>

<item 33>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“Thanks to the First Amendment, Florida and Texas cannot force websites or social media apps to host hateful content, misinformation and spam, as their deeply
misguided laws would require,” Oregon Senator Ron Wyden, who co-authored Section 230, a law that protects social media companies’ content moderation decisions,
told TechCrunch. “… A ruling in favor of the Texas and Florida laws would create utter chaos, and make many sites worthless to regular users who want to watch a
funny video or see family photos.”

NetChoice President Steve DelBianco also warned that allowing the state laws to go into effect would unleash “a tidal wave of offensive content and hate speech
crashing onto users, creators, and advertisers” that would force Americans to wade through “racial epithets, aggressive homophobia, pornographic material,
beheadings, or other gruesome content” just to use social apps.
</item 33>

<item 34>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Many would say the writing is on the wall for Twitch, after endless headlines about losing top talent, poor management, bad platform policies, and creator
burnout. But the idea that Twitch has lost its way after an ostensible golden era is an easy narrative that misses something much starker. If Twitch is in
trouble in 2023, it’s not because things have gotten worse. Everything you’re seeing is the natural progression of an unsustainable system. The prognosis has
always been terminal; we’ve just refused to see it because it’s been profitable.

It didn’t take long for publishers to see that Twitch could make or break a game. Around 2018, Twitch and its personalities turned Fortnite into a juggernaut so
powerful, much of the gaming industry started molding itself in its image. While an approachable aesthetic and surprising gameplay made for good viewing, what
kept Fortnite from being a mere fad was the model. Before, content expansions and adjustments could take months, if not years, to hit, and you might have to
shell out money to see any of it.
</item 34>

<item 35>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
And if we can solve it, it will do so much good. It really, really will because we will find out we have more in common than not,” he said.
</item 35>

<item 36>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
Not enough conversation.”

There were other things that may have hurt Pebble’s traction, too, including its lack of a native mobile app. As T2, the startup prioritized building for the
web to be agile and more flexible, but lost out on the opportunities for discovery driven by the app stores. Notes Greer, “that might have been a mistake.”

In retrospect, Cselle also questions if the rebranding from T2 to Pebble may have been an issue.

“I’m wondering if the Pebble rebrand had actually softened the value prop[position] because T2 just encapsulated it so much — that it was really duplicative [of
Twitter],” he says.

Like most startup shutdowns, the answer as to “why” is not any one thing but a combination of factors. For Pebble, that’s certainly true.
</item 36>

<item 37>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“They’re in the same ballpark, the sentiment is the same,” Barrett said. “Republican lawmakers in each state believe that — and said explicitly in the course of
debating and passing these laws — that ‘Silicon Valley oligarchs’… are ideological liberals, and they are censoring people in our states who are conservative,
and we are hereby ordering them to stop doing that.”

How could the Supreme Court decision affect social media companies?

If the court finds that social media companies don’t have a First Amendment right to curate the kinds of content that they allow, social platforms could look
very different, at least in states that are trying to limit their moderation powers. After years of slow progress on misinformation — and worrisome backsliding
on platforms like Elon Musk’s X — the Supreme Court’s decision could upend that process, sowing chaos online in the process.
</item 37>

<item 38>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think Threads and others have encountered this as well,” adds Pebble co-founder and CTO Michael Greer, previously Discord engineering head. “The durability
of the network effect of Twitter is stronger than anticipated,” he says.

Despite these challenges, Pebble initially had decent retention. By week four, it was seeing retention rates of 30%, in fact. Its invite list was also working,
as 60% of people who received a Pebble invite converted to become users. In addition, some 10,000 users arrived at that list from early press, like TechCrunch’s
coverage of its first outside funding — a $1.1 million angel round that included investors like former Google VP Bradley Horowitz, Android co-founder Rich Miner
and the former CEO of Wikipedia, Katherine Maher. Data from Similarweb indicates the Pebble website topped 99,300 visits last month, indicating repeat usage.

The company’s hypothesis was that consumers were hungry for a Twitter alternative that prioritized trust, safety and moderation from the start.
</item 38>

<item 39>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 39>

<item 40>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
But as for social media as a whole, of which Reddit would presumably play a part, Ohanian said it brings out the “worst parts of our nature in many ways.”
What’s more, he expressed worries about the desire for people to be first and win on whatever leaderboard a platform features, which then leads to a situation
where misinformation is able to spread.

Humans evolved trusting the faces & voices we knew around the campfire. The idea of believing something bc of experts/institutions you've never met is actually
new to our species. I visited @UVADemocracy + chatted with @karaswisher + @dkroy about what's next for social media. pic.twitter.com/PMBXi7T3vv — Alexis Ohanian
🇦🇲 (@alexisohanian) October 31, 2023

“Now, there’s so much pressure to be first, and UGC — user-generated content — will always win. It won’t always be right, though.
</item 40>

<item 41>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Does Reddit co-founder Alexis Ohanian have regrets over his creation? The entrepreneur and founder of venture fund Seven Seven Six, speaking at an event at the
University of Virginia, espoused a fairly negative view of the current social media landscape and the creator economy — an environment where he says, everyone
has “been so trained and conditioned now to just get the likes and retweets, get the upvotes…I wish I could be more optimistic about it, but we’re seeing it
play out before our eyes.” Still, he ultimately described himself as a “techno-optimist” — an apparent nod to a16z General Partner Marc Andreessen’s recent
diatribe where he claimed technology could solve all our problems.

“Guilty as charged,” Ohanian said, during his interview with Kara Swisher. “I do think that pendulum will swing back. I think even swifter than we realize.”
</item 41>

<item 42>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Even if Jake Paul was less despised online, endorsements aren’t as valuable as they used to be. A paper published in the International Journal of Communication
this year reported that mobilizing influencers to build support for political causes is more effective than using their endorsements to gain votes. Blatant
endorsements lack personal connection, and create an “authenticity gap.” Authentic creators are trustworthy, and in campaign strategy, that’s more valuable than
the size of their following.

Martin Riedl, a University of Tennessee, Knoxville professor who studies social media and co-authored the paper, said that society tends to reward public
figures for speaking out on social justice issues because there’s more “situational awareness.”

“If you use influencers in your campaign, it’s important that they can authentically believe in what you promote,” Riedl said. “And if that’s not the case, that
makes it really difficult for people to take your campaign seriously.”

Authenticity reigns

Facebook ads, celebrity endorsements and campaign trail rallies aren’t enough to engage voters online.
</item 42>

<item 43>
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
</item 43>

<item 44>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Sunfolding

Is reportedly winding down its operations. On its website, Y Combinator categorizes Sunfolding as an “inactive” company.

Zulily

Will lay off more than 500 employees after closing fulfillment centers in Ohio and Nevada in February 2024.

D2iQ

Is shutting down its operations and selling its assets to Nutanix.

Atmosphere

Has reportedly eliminated a large part of its workforce. The business-focused streaming TV service raised $100 million in 2022.

Tidal

Is laying off more than 10% of its staff, the company confirmed to TechCrunch December 7.

ZestMoney

Is shutting down operations. The Goldman Sachs-backed “buy now, pay later” startup was once valued at $450 million.

Navan

Is cutting about 145 roles as the travel startup previously known as TripActions plans to become public.

Incredibuild

Is laying off 20% of its workforce. The software development startup previously raised $35 million in a Series B funding in 2022.
</item 44>

<item 45>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
For background, the company announced its plans to enter the fediverse in December 2022 and this May launched a private beta for Mozilla.social, a Mastodon
server — or “instance” in fediverse parlance — that will allow consumers to participate in this new form of social networking where problems like
misinformation, harassment and issues around worsening mental health are minimized.

The company’s work on its social efforts is also available publicly on GitHub. (Of interest to technical folks, the company is using a forked version of Elk as
its alternative web client for its Mastodon server, which is what gives it its slick look and feel.)

Mozilla’s involvement, says O’Hara, came about because the company looked at the history of social media over the past decade and didn’t like what it saw.

“I think that it’s a pretty poor track record by existing companies that are only model motivated by profit and just insane user growth, and are willing to
tolerate and amplify really toxic content because it looks like engagement,” she says.
</item 45>

<item 46>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
In the years since, Democratic campaigns have increasingly prioritized social media strategy, establishing politician-influencers who wield content for votes.

A 2022 midterm report by the Alliance for Securing Democracy found that in the Senate races, 47% of Democratic candidates had TikTok accounts, compared to 12%
of Republican candidates. Of the major party House candidates, 30% of Democrats and 10% of Republicans were on TikTok, which as a platform was most prevalent in
gubernatorial races.

Though opposition to TikTok has been bipartisan, the crusade to ban the platform over national security concerns is primarily led by Republicans at the state
level — a position that likely factors into the party’s social media struggles with young voters.

Mehmet Oz, one of the few Republican candidates on TikTok, had the advantage of already having a large social media following when he announced his candidacy
thanks to his popularity as Dr. Oz.
</item 46>

<item 47>
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
</item 47>

<item 48>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
I think even swifter than we realize.” The war in the Middle East may even be the tipping point, he theorized, as it’s now “abundantly clear” that we need
something better to figure out the world than the platforms we have today.

The founder, as you may recall, notably resigned from Reddit’s board in 2020 over differences of opinion on moderation and wanting to crack down on communities
about violence and hate. It was a moment where he seemed to be reckoning with what his platform had wrought. At the time, he asked for his board seat to be
replaced by a Black board member. The company agreed, appointing its first Black board member with Y Combinator partner Michael Seibel.

Reddit also later banned several toxic communities, including r/donaldtrup following violence at the U.S. Capitol, and Ohanian says that the site has “gotten
demonstrably better,” since, as well as better as a business.
</item 48>

<item 49>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Google is now making it the default sign-in method for all users.

2 sec, let me text you some cash: When questioned about Mastercard’s prospects in emerging markets such as India, Mastercard’s CFO Sachin Mehra praised UPI for
helping with digitization but voiced reservations about its commercial sustainability, saying it is an “incredibly painful experience” for ecosystem
participants.

Bravely browsing — or searching — for a new job . . . : Brave Software, the maker of Brave Browser and Search, confirmed that it has laid off 9% of its
workforce across departments.
</item 49>

<item 50>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
There’s this picture from more than a decade ago that still goes viral on the web every once in a while. You’ve probably seen it: it was created by the venture
capitalist Andrew Parker, and it compares a few dozen startups with subsections of Craigslist. Back then, Craigslist was all things to all people online, and a
generation of startups figured they could do part of the job a lot better.

Some of those companies failed — sometimes because of Craigslist! — but some also became gigantically successful. Airbnb worked because it was more searchable,
reliable, and trustworthy than a random Craigslist listing; StubHub sold you secondhand tickets without the 50 percent chance those tickets were fake; Etsy
offered a much more fun shopping and discovery experience than a bunch of text listings and crappy photos.

For a decade, startups tried to unbundle Craigslist. Now they can do the same for social networks.
</item 50>

<item 51>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
Forget the hand-wavy protocol stuff for a second — one of the best things about embracing ActivityPub is that it sticks a crowbar into a single Voltron-ic
product like Facebook or Twitter or Snapchat and pries it apart into its component pieces, each one ripe for innovation and new ideas.

We’re still in the very early days of the fediverse, and it’s going to be messy for a while. It might feel like you’re seeing the same posts too many times, and
like you see some posts that obviously weren’t meant to be seen in the app or feed you’re using. Some particularly thirsty influencers are going to go hard on
cross-posting tools that threaten to clutter up all your feeds everywhere and become totally unavoidable. This is not a problem with the protocols; it’s an
opportunity for better products. There’s plenty of money to be made in the fediverse, and plenty of space for new products to take off.
</item 51>

<item 52>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 52>

<item 53>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Though he’s glad that access to the technology has broadened, that has come with many complications.

“I don’t like the idea that the internet I’m so nostalgic for was actually an ivory tower,” he said, but on the flip side, he said, we’re all exposed to more
people, in real-time, than we could have ever fathomed. And that can put people at odds with one another and with everyone’s different versions of the truth —
even if their truth is believing in something as far-fetched as the earth being flat, he said, pointing to one example that has a following on Reddit.

Meanwhile, asking platforms like Facebook, Twitter, or Reddit to be in charge of verifying truth and accuracy is a difficult task, he said. Ohanian also called
out TikTok as “an arm for the Chinese government” that’s really good at “feeding us what we want, when we want it” — in other words, not always a fountain of
truth, either.
</item 53>

<item 54>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
“We are saying out the gate that this isn’t a neutral platform,” O’Hara stresses. “We think that that’s often used as this crutch to allow, or even amplify,
really toxic content in the name of engagement… In some cases, platforms aren’t just brave enough to actually just take things down.”

Building directly on the fediverse isn’t the only way the company is prompting new forms of social networking. The company has also financially backed a
startup, Mammoth, the makers of a third-party app for Mastodon.

Mozilla has been leaning into its values around inclusion, dignity, security, agency and community for 25 years, O’Hara says, and those will guide its efforts
in the fediverse, as well.

“We’re going to have rules on how people can engage with one another…it’s the fediverse, you have lots of other places to go if you don’t want to abide by those
rules,” she adds.

Users can provide Mozilla with feedback via @Social @Mozilla.social or by using the hashtag #mozillasocial within the product.
</item 54>

<item 55>
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
</item 55>

<item 56>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Aside from forcing platforms to allow some forms of content that would otherwise be disallowed, these laws also seek to force social media companies to provide
users with individualized explanations when their content is removed or restricted. Because this process is largely conducted algorithmically now — generally
with light human intervention or oversight — social media companies might need to reimagine their content moderation systems or hire way more humans to respond
to these incidents. Those kind of adjustments would likely be costly and difficult to scale.

“As you can imagine, that can get kind of onerous when you’re taking down millions and millions of pieces of content a day, and much of that activity of the
vast majority of it is currently being done automatically,” Barrett said. “The idea that a human being would have to go back and explain each time something
came down would be quite a challenge.”
</item 56>

<item 57>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
Instagram Threads isn’t the only app joining the decentralized social web, which includes Twitter/X rival Mastodon and other apps, with its newly added support
for the networking protocol ActivityPub. Today, the social magazine app Flipboard is announcing it has also now integrated with ActityPub. In its initial phase,
select Flipboard accounts will be discoverable and can be followed by the millions of users of decentralized social apps, including Mastodon. Over time, all
profiles on Flipboard will be available in the fediverse, as this network of decentralized social apps is known.

The company announced its intention earlier this year to participate in the fediverse. It began by integrating its app with Mastodon via an API and setting up
its own Mastodon server, flipboard.social, ahead of full ActivityPub integration. This allowed Flipboard to get a feel for the world of decentralized social
media, and learn how its users would respond.
</item 57>

<item 58>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Despite the problems with social media and the urgency, in terms of its impact on society, Ohanian believes there’s still some hope. He noted, for example, that
the founders his fund 776 invests in are much smarter and more aware of the issues than he ever was at their age. Perhaps the next platforms will still be
addictive, but in a way that doesn’t “further disrupt democracy,” he mused.

He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not. “I think we’ll
see a rebirth of live gatherings and in-person experiences…the bright side of this is that I hope it then gets us back in touch with our humanity, and the
things that no amount of AI can replace.”

“I want to be an optimist about it,” Ohanian added. “And I think there’s still a path forward but we’re fighting against very, very strong human things.
</item 58>

<item 59>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation. Its policies have resulted in increased hate speech
and toxic content on the platform, some studies have shown.

Mozilla believes there’s a better way forward and, more importantly, that now is exactly the right time to invest in that path. And, as a wholly owned
subsidiary of a nonprofit, the company says it’s not motivated by generating earnings for shareholders or returning a VC investment, allowing it to progress
with a collaborative approach where it takes in input from a lot of different voices.

In an interview with TechCrunch, Mozilla Senior Director of Content Carolyn O’Hara explained why Mozilla has taken an interest in the fediverse and Mastodon,
specifically, and how it expects to experiment in the months ahead.
</item 59>

<item 60>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
It also gave Flipboard a way to stay connected to social media after Twitter/X increased its API fees for third-party developers, making it unsustainable for
many developers to continue to work with the company.

The Flipboard app’s primary purpose has been to curate into social “magazines” news and information found on the web, including links to articles, photos and
other social posts. As a result, it had relied on Twitter as one of its sources of information. That changed this year, when Flipboard shifted its Twitter
integration over to Mastodon and another alternative social app, Bluesky. It also set up its own Mastodon server and began curating news across the fediverse
via editorial “desks” focused on improving the discovery of news on Mastodon.

All this was in the lead-up to making Flipboard itself a federated social app, a process that’s kicking off today.
</item 60>

<item 61>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
Brands, too: Remember the Tony the Tiger VTuber in August of 2022?

Many creators that jumped on the trend have since stopped using their avatars, but at the start of the COVID-19 pandemic, there was a large increase in VTubers
on Twitch and other streaming platforms, like YouTube, as models and equipment became more accessible to the general public. Twitch has since gone on to
highlight a wide variety of VTubers through its VTuber Takeover promotion, which most recently ran in August. Twitch also reported that VTubing-related content
on the platform shot up by some 350% between January and August 2022. Despite this, some have argued that the VTubing bubble has all but burst.

Polygon is celebrating TwitchCon 2023 with a package of stories covering streaming trends and news from the show. Check it out!

“I do feel that VTubing has become less novel as the market has become more saturated,” says a streamer with agency Nijisanji who is not approved to speak
publicly.
</item 61>

<item 62>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
In a note to employees Monday, Spotify founder and chief executive Daniel Ek — citing slow economic growth and rising capital costs — said right-sizing the
workforce is crucial for the company to face the “challenges ahead.”

TuSimple exits: When TuSimple went public in 2021, it was flying high as the leading self-driving trucks developer in the U.S. Now — after a string of internal
controversies and the loss of a critical partnership with truck manufacturer Navistar — TuSimple is exiting the U.S. altogether. TuSimple said in a regulatory
filing Monday that it’s laying off the majority of its U.S. workforce and selling assets here as it exits the country for Asia.

ZestMoney shuts down: ZestMoney — a buy now, pay later startup whose ability to underwrite small-ticket loans to first-time internet customers attracted many
high-profile investors, including Goldman Sachs — is shutting down following unsuccessful efforts to find a buyer.
</item 62>

<item 63>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
Right now, every social platform is a universe unto itself. The only way to see your Facebook posts is to sign up for Facebook’s rules, Facebook’s aesthetic,
Facebook’s features, Facebook’s moderation, Facebook’s algorithm, and Facebook’s business model. That’s not just a bad user experience, it’s a ridiculous way to
run a company. Facebook has to invent and maintain all these things, just as every one of its competitors does. It’s an impossible task but just as impossible
to compete with; you can’t build social products without building an entire social graph from scratch, and good luck with that.

In a fediverse world, rather than try to build another all-encompassing Facebook from scratch, an enterprising developer can pick one or a few of its features
and try to do better. Users can pick their favorite app — or two, or two hundred — through which to get their posts.
</item 63>

<item 64>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
“What are the implications of federation in the front end? How do we think about curation and all the things, all the capabilities and tools that we’ve created
over the years? How does that work in a world that’s federated, and from a user experience point of view? That is a great question,” he said.

Despite all the changes, Flipboard is not in need of raising funds to support its new developments. It’s running off the profits of its own business as it moves
in this direction.

The company is also betting on the fact that federated social media may only be the beginning of what’s to come for the web overall.

“I saw what was happening with ActivityPub and it became very clear to me that this is the future of the web, period,” McCue said. “The social web is people
linking to pages and people linking to people. So it’s a much more intricate web.”

He sees Flipboard as a part of that opportunity.
</item 64>

<item 65>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
At midnight last Friday, Alphonzo Terrell held up a glass of Dom Pérignon and toasted his one-year anniversary of being laid off at Twitter.

If it weren’t for that moment, he wouldn’t be in Austin at an AfroTech after-party, surrounded by thousands of people who now know his name and his social media
app, Spill. As a competitor to the platform now known as X, Spill hails itself as a safer place for Black Americans and the LGBTQ community and has amassed
around 200,000 users, he said. This past year, the app, which is still in beta mode, has raised a total of $5 million in pre-seed funding, including a recent $2
million extension round led by Collide Capital.

“We put an invite-only beta in June, and within three weeks, it went viral,” Terrell told TechCrunch.
</item 65>

<item 66>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 66>

<item 67>
How the conspiracy-fueled Epoch Times went mainstream and made millions (Hacker News, 2023-10-16T03:41:24+00:00)
It wasn’t wrong, he said, to favorably cover a politician who shared Falun Gong’s conservative values and whose goals aligned with their own.

“If someone comes along now who can help to halt the downward spiral that the world is in, then he is truly someone extraordinary!” Li said. “He would in effect
be helping us! Wouldn’t he be helping us to save people?”

The Epoch Times also employed a tactic more often associated with fake news content farms and scammers than news organizations, creating a network of inspiring
and cute-content pages and fake accounts to inflate The Epoch Times' reach.

In 2019, following reporting by NBC News, Facebook found that The Epoch Times had “leveraged foreign actors posing as Americans to push political content” and
banned it from future advertising, citing a violation of policies, including trying to circumvent review systems.
</item 67>

<item 68>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
Another app that spun out from laid-off former Twitter employees, Pebble (formerly T2), recently shut down after a year, growing to just 20,000 registered
users. Its CEO, Gabor Cselle, previously had exits to Google and Twitter, but accepted that Pebble simply wasn’t competitive enough to stay afloat. But while
Pebble sought to essentially clone Twitter, Spill’s team is building something that looks very little like it.

“People are looking for something new,” Terrell said. “I think things that have really defined, unique value propositions are going to win over the long term —
it might not be like there’s one winner-take-all.”

Earlier this year, TechCrunch covered the founding of Spill, piquing the interest of Kapor Capital, which ended up writing the company its first check.

“We had over 25,000 people join our waiting list in less than 24 hours from that article,” Terrell said.
</item 68>

<item 69>
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
</item 69>

<item 70>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
“The [World Health Organization] now calls it the loneliness epidemic. And if they say that it’s an epidemic, it’s because it’s actually infectious. In other
words, if you’re isolated, your loved ones are too because you’re unreachable. So during the two hours you’re on TikTok, they have no one to talk to,” Martin
told me.

“And at the same time, the human needs that the social consumer space can fulfill are no longer covered by these products, whereas they used to be,” he added.
“In the early days of Facebook, I don’t know if you remember, profiles were sort of a mash-up. There were drawings, games, photos, text. You’d write long
comments, it could be a poem . . . And on the other hand, it was a reminder that you mattered to these people.”

According to him, the current generation of social networks are very passive.
</item 70>

<item 71>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
He said he never conceived of the influence the site would one day hold — when Reddit became a multibillion-dollar business with hundreds of millions of users.
That thought would have been “ludicrous,” he said. And he certainly never could have imagined his startup could have become so successful that it could affect
elections and democracies.

“To… think Oh my god, I’m gonna be so successful, it could have a tremendous impact on democracy would take a level of delusion that even I, as a first-time
CEO, just couldn’t have,” he noted. “Was that a blind spot? Absolutely,” he admitted.

Ohanian identified that another part of the problem is that there are more people online today than early tech founders may have imagined. Though he’s glad that
access to the technology has broadened, that has come with many complications.
</item 71>

<item 72>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“Gen Z does not put up with bullshit,” Henry said. “That’s so much of what platforms like TikTok and Twitch cater to, and why they thrive on those platforms,
because you can just pull out your phone and be talking while you’re getting ready with the worst angle possible. Republicans in general, it goes against so
much of how they act otherwise and young people know that.”

Ramaswamy’s TikTok strategy is puzzling; he appears to be trying to replicate the success of previous candidates who became social media stars over the course
of their campaigns, but his TikTok presence conflicts with his own stances on social media and young voters. His communications director, Tricia McLaughlin, did
not immediately respond to TechCrunch’s request for comment.

Like many of his Republican colleagues, Ramaswamy has accused TikTok of being a threat to national security, and during a town hall days before he launched his
account, described the platform as “digital fentanyl” from China.
</item 72>

<item 73>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
I’d bet heavily that ActivityPub becomes the default choice over time, but ultimately, it doesn’t matter so much which protocol wins as long as one of them
does. We don’t need two internets, and we don’t need two social protocols. We need one thing that is both simple enough and big enough to handle all the ways we
connect with each other online. No centralized platform has ever been big enough. The fediverse can be.

Mastodon is the biggest thing in the fediverse so far, but it may not be for long. Image: Mastodon

Decentralizing social media can sound like a sort of kumbaya anti-capitalist manifesto: “It’s about openness and sharing, not capitalism, man!” In practice it’s
the opposite: it’s a truly free market approach to social networking. Mastodon may not be interested in becoming a trillion-dollar company, but there’s no
reason there can’t be plenty of those built on the fediverse.
</item 73>

<item 74>
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
</item 74>

<item 75>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
13; Sardine, No. 16; and Tapcheck, No. 48.

CB Insights also released its Fintech 100 list, made up of the “most promising 100 fintech startups of 2023.”

Other items we are paying attention to:

The great Zelle pool scam

The Public platform offers everyday investors a slice of ‘Shrek’ catalog with quarterly payouts

Stockpile and Green Dot partner on debit cards for minors

Amex pilots biometrics in online checkout

Pie Insurance appoints Audra Foglietta as chief financial officer

Monzo picks Cash App vet Conor Walsh for US CEO

Folsom-based EV Life launches electric car loan to lower monthly payments

CLEAR launches reusable KYC solution

Funding and M&A

As seen in TechCrunch

Open banking led to a fintech boom — as Brite raises $60M, account-to-account payment grows

Recapitalization,
</item 75>

<item 76>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
– Jonathan

Signing off

This weekend, pour one out for Tumblr, one of the most interesting social networks on the internet, which appears to be in trouble. It’s not dying, but it’s
not… not dying, you know? Whatever happens next, this state of affairs is a bummer for a lot of reasons, including that Tumblr promised to work with ActivityPub
— which would have been a big win for the fediverse — and I think is still maybe the web’s best and most versatile posting tool. Matt Mullenweg, the CEO of
Automattic (which owns Tumblr), spent a bunch of time this week answering people’s questions about the future of Tumblr, and it paints a sad but interesting
picture of what it really takes to build a better social network. It’s all making me root for Mastodon even harder.
</item 76>

<item 77>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 77>

<item 78>
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
</item 78>

<item 79>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Ken Russell, a Democrat who left the Miami City Commission to run for a House seat in 2022, leaned into the cringe with bait-and-switch thirst traps reminding
viewers to vote. In another video titled “Appealing to the youth vote,” he recreated Steve Buscemi’s “How do you do, fellow kids?” to encourage voter
registration.

Audience engagement doesn’t rely on forcing fun. North Carolina Rep. Jeff Jackson updates constituents on TikTok, breaking down topics like the government
shutdown in concise explanatory videos without the frills of internet humor. Even though his content is less exciting, his account has over 2.2 million
followers.

As an alternative to the politician-influencer, some campaigns are focusing on mobilizing creators who already have an engaged audience. Biden is not on TikTok,
but his digital strategy team is building an “army of influencers” to reach viewers who wouldn’t typically keep up with the White House press corps.
</item 79>

<item 80>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
To that end, Pebble’s founding team included co-founder Sarah Oh, Twitter’s former human rights advisor.

Pebble continues to believe its approach to moderation was correct, even if it was not ultimately a factor in driving growth.

“We came in with a particular angle: kinder, safer. Trust and safety. And I think our approach to moderation that we did have on the site did work,” says
Cselle, adding that Pebble didn’t have some of the issues other federated platforms faced. For example, TechCrunch reported that Bluesky earlier this year had
strained its relationship with Black users, as it failed to crack down on bigotry on the site. Later, users on Bluesky were even creating usernames with racial
slurs, and yet the platform didn’t issue a public apology. Pebble never faced any such controversies.

But it may have positioned itself a little too far into the “kindness” territory, the founders now believe.
</item 80>

<item 81>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
This comes after its first round of layoffs, which eliminated 150 roles about a year ago.

Shopify

Announced May 4 that it is laying off 20% of its workforce, impacting more than 2,000 people. It’s also selling its logistics business to Flexport for roughly
13% in stock.

Bishop Fox

Announced May 3 that it laid off around 50 employees — or 13% of its workforce — on May 2.

Neato Robotics

Announced May 1 that Vorwerk-owned Neato Robotics is shutting down, with nearly 100 employees impacted by the move.

April 2023

Clubhouse

Announced April 27 that it has laid off more than 50% of staff. A spokesperson for Clubhouse declined to comment on the number of people impacted by today’s
workforce reduction or the number of employees who remain at the company. Last October, Davison told TechCrunch that Clubhouse had close to 100 employees.
</item 81>

<item 82>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Clearco

Announced on January 17, impacting 30% of staff across all teams.

ShareChat

Announced on January 15, ShareChat laid off 20% of its workforce — or over 400 employees — just a month after eliminating more than 100 roles.

SmartNews

Announced on January 12 a 40% reduction of its U.S. and China workforce, or around 120 people.

Intrinsic

Alphabet’s robot software firm, Intrinsic, is laying off 40 employees TechCrunch confirmed on January 12. Amounting to around 20% of the headcount.

Greenlight

The fintech startup offering debit cards to kids laid off 104 employees on January 12, or over 21% of its total headcount of 485 employees.

Career Karma

Learning navigation platform Career Karma laid off another 22 people on January 12 across its global and domestic workforce.

DirectTV

Announced on January 12 plans to lay off about 10% of its management staff on January 20.
</item 82>

<item 83>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 83>

<item 84>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
It’s just that in a fediverse-dominated world, the way to win is not to achieve excellent lock-in and network effects. The only way to win is to build the best
product.

This is really not a particularly hot take, by the way. Even the most successful centralized platforms have long understood that a protocol-driven social web is
a good idea. Jack Dorsey used to say that Twitter was better as a protocol than a platform, and started the project that became Bluesky before also helping get
Nostr going. (We really don’t need to get into the whole story of what happened to Twitter since then, except to say that the speed with which that platform
changed made a lot of people acutely aware that we need a social ecosystem that can resist the whims of a single company or CEO.)

Meanwhile, Meta CEO Mark Zuckerberg seems to earnestly believe that bringing Threads to the fediverse is a good idea, both for its product and for its business.
</item 84>

<item 85>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A comprehensive list of 2023 tech layoffs From major layoffs at Google, Amazon and Microsoft to small fintech startups and apps

Last year’s techwide reckoning continues. The tech industry has seen more than 240,000 jobs lost in 2023, a total that’s already 50% higher than last year and
growing. Earlier this year, mass workforce reductions were driven by the biggest names in tech like Google, Amazon, Microsoft, Yahoo, Meta and Zoom. Startups
across many sectors also announced cutbacks through the first half of the year. And while tech layoffs slowed down in the summer and fall, it appears that cuts
are ramping up yet again.

Many economists have cautioned against fears of a recession, which would seem like a reason for optimism. But the momentum for a tech sector rebound has been
slow to build, resulting in tech companies continuing to cut back on their workforces and pivot from a growth mindset to one based on efficiency in the face of
stubborn market conditions.
</item 85>

<item 86>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
But tracking these layoffs helps us to understand the impact on innovation, which companies are facing tough pressures and who is available to hire for the
businesses lucky to be growing right now. Unfortunately, it also serves as a reminder of the deeply human impact of layoffs and how risk profiles could evolve
from here.

Below you’ll find a comprehensive list of all the known layoffs in tech that have occurred in 2023, to be updated monthly. If you have a tip on a layoff,
contact us here. If you prefer to remain anonymous, you can contact us here.

The running total of layoffs for 2023 based on full months to date is 224,503, according to Layoffs.fyi. Tech layoffs conducted to date this year currently
exceed the total number of tech layoffs in 2022, according to the data in the tracker.

Data visualization by Miranda Halpern, created with Flourish

December 2023

ShareChat

Is cutting 15% of its workforce in a “strategic restructuring” effort for 2024.
</item 86>

<item 87>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 87>

<item 88>
Flipboard becomes a federated app with support for ActivityPub (TechCrunch, 2023-12-18T16:00:49+00:00)
He sees Flipboard as a part of that opportunity. “There needs to be a way to do discovery and search and have it be beautiful and simple and easy to use. That
is what we’re focused on,” McCue added.
</item 88>

<item 89>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
It’s difficult to extrapolate from the 2015 and 2020 numbers to today’s (which, as we have seen from the evidence presented here, will almost certainly not be
the whole story), but Bonta noted that the large figures are presented for impact, not as legal justification.

“The basic premise remains that their social media platforms are used by millions of children under 13. Whether it’s 30 percent, or 20 or 10 percent… any child,
it’s illegal,” he said. “If they were doing it at any time, it violated the law at that time. And we are not confident that they have changed their ways.”

An internal presentation called “2017 Teens Strategic Focus” appears to specifically target kids under 13, noting that children use tablets as early as 3 or 4,
and “Social identity is an Unmet need Ages 5-11.”
</item 89>

<item 90>
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
</item 90>

<item 91>
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
</item 91>

<item 92>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
According to him, the current generation of social networks are very passive. You don’t have to do much to spend two hours on TikTok or YouTube because these
companies want you to spend as much time as possible in these apps. “We aspire to go back to these earlier precepts and make them work,” Martin said.

That’s also the reason why Amo doesn’t want you to spend hours in the app. When you have a few minutes, you can open the app to check what’s new on your
friends’ profiles by swiping up on the notification cards.

When you’ve reached the last card, ID shows you a message that says “get some fresh air.” And then, the app closes itself. You’re back on the home screen, you
can put your phone back in your pocket.

Amo & ID

ID is an opinionated take on social apps, but is it going to work?
</item 92>

<item 93>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
“Not everyone wants to use one product,” he said to my colleague Alex Heath earlier this fall, “and I think making it so that they can use an alternative but
can still interact with people on the network will make it so that that product also is more valuable.” Mark Zuckerberg! Said that! He understands that people
want to feel like their social connections and content belong to them, not to a company that can disappear or pivot or change its ways. And let’s be real: if
Meta can’t build and maintain the One True Social Network For Everybody, nobody can. It’s well past time to try something else.

The only way to see your Facebook posts is to sign up for Facebook’s rules, Facebook’s aesthetic, Facebook’s features, Facebook’s moderation, Facebook’s
algorithm, and Facebook’s business model.

Right now, every social platform is a universe unto itself.
</item 93>

<item 94>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
To explain the Supreme Court’s decision to wade into these issues — and what happens next — TechCrunch spoke to Paul Barrett, NYU adjunct law professor and
Deputy Director of NYU Stern’s Center for Business and Human Rights.

Why is the Supreme Court involved?

These cases actually started in Florida and Texas a few years ago before wending their way to the Supreme Court this year. In those two states, Republican
lawmakers passed parallel laws to control how social media companies operate. In Florida, Governor Ron DeSantis signed Senate Bill 7072 into law in May 2021. In
Texas, House Bill 20 made it through the state legislature and was signed by Governor Greg Abbott in September 2021.

“The reason why these cases are in front of the Supreme Court is actually relatively simple: Florida and Texas were more or less the first out of the gate in
imposing this type of restriction on social media companies,” Barrett explained.
</item 94>

<item 95>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
“Suddenly, we’re supporting a city, we have data costs, we have infrastructure, we have to accelerate our building times, so we really needed an extension to do
that effectively.”

His journey with Spill started last year at AfroTech. He arrived at the conference just days after being laid off from what was then Twitter, along with about
3,000 others amid Elon Musk’s acquisition. Terrell hopped on a call with former Twitter colleague DeVaris Brown about a business idea and spent three days in an
Airbnb creating the pitch deck for what became Spill. At AfroTech, he went around taking last-minute coffee meetings, seeing who would give his company a
chance. Flash-forward to this year and Spill is the cool kid in town.

So many apps have sprung up over the last year in an attempt to become “the new Twitter,” but no app has emerged as the dominant refuge for disillusioned
Twitter users.
</item 95>

<item 96>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The latest funding round brings QI Tech’s total capital raised to $262 million.

New Web Summit CEO appointed: Former Wikimedia Foundation CEO Katherine Maher has been appointed as the new CEO of Web Summit, following the controversial
departure of its previous leader, Paddy Cosgrave. Maher’s appointment comes as the tech conference faces a challenge to rebuild relationships and trust within
the tech community.

Top reads on TechCrunch this week

Shinier, smarter maps: Google is rolling out a variety of updates for Maps, incorporating AI technology to enhance user experience. The feature is currently
available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.

From stop-and-start to charging ahead: Over on TC+, Tim has had it with U.S. automakers hesitating on EVs, as Ford and GM are showing signs of backing away from
their commitment to electric vehicles. Cowards.
</item 96>

<item 97>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
For Pebble, that’s certainly true. It was a perfect storm of competition, X’s continued traction, the lack of a native app, a brand that didn’t resonate and a
space that was maybe a little too safe to be as addictive and as fun as the original.

With Pebble’s winddown, being announced to users today at 9 AM PT, early adopters will have the option of exporting their Pebble archive as a zip file that
loads a mini web page showcasing all their old posts. Pebble won’t be directing users back to X however, or any other social network.

As for where the founders will now hang out?

“I don’t know. Maybe LinkedIn,” Cselle jokes.

Pebble is also returning a small amount of the funds left over to its investors, more as a courtesy to show financial responsibility. The Pebble website will be
shut down on November 1st and there aren’t further plans for the IP at this time.
</item 97>

<item 98>
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
</item 98>

<item 99>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“The amount of media and content that we consume, political and cultural, where people get their information … It’s getting more and more intertwined,” Henry
said. “To be effective ideologically, when it comes to campaigns and these wins that we want, we have to understand how these ecosystems work, and be
strategically using them or else we risk being left behind.”

Engaging with voters online requires more than just working with popular creators. In Ramaswamy’s case, an endorsement from Jake Paul only drove animosity from
TikTok viewers. While his older brother, Logan Paul, has made the occasional social commentary on his podcast, the younger Paul has rarely engaged with politics
or social issues. Jake Paul is generally disliked online for the running list of allegations against him, including sexual assault accusations.

The Paul brothers have a reputation of promoting anyone and anything for a paycheck, and when Jake Paul posted a political endorsement out of the blue, viewers
immediately questioned how much Ramaswamy paid him.
</item 99>

<item 100>
2023 in social media: the case for the fediverse (The Verge, 2023-12-19T13:00:00+00:00)
There are already a few platforms built on ActivityPub and embracing the ideas of the fediverse. Apps like Mammoth and Ivory are showing the potential for
different user experiences on top of the same data and infrastructure. But so far we’re mostly in the “popular app, but federated” phase of this transition.
(Which is definitely better than “popular app, but blockchain” from a couple of years ago… but not by much.)

So far we’re mostly in the “popular app, but federated” phase of this transition

Mastodon deliberately looks and feels like Twitter. (So do Firefish and Pleroma and GoToSocial and others. The fediverse is super into replacing Twitter.)
Pixelfed is Instagram through and through. Lemmy’s features are virtually all ripped straight from Reddit. Almost everything in the fediverse is a one-to-one
competitor to an existing platform: PeerTube to YouTube; Friendica to Facebook; BookWyrm to Goodreads; on and on it goes.
</item 100>
=== END ITEM id=4dd04af83e70f994 ===

=== ITEM id=9483edeee994808f ===
Gold sentences and facts:
Sentence 0: Likely with the sole intent of keeping Messi fit in that time off, and the added bonus of raking in significant cash, Inter Miami scheduled a two-match tour of China in early November after their regular season concluded.
  Fact 0.0: Inter Miami scheduled a two-match tour of China in early November.
  Fact 0.1: The China tour was scheduled after Inter Miami's regular season concluded.
  Fact 0.2: The likely sole intent of the China tour was keeping Messi fit during that time off.
  Fact 0.3: The China tour had the added bonus of raking in significant cash.
Sentence 1: Eventually, Alvaro Barreal’s 78th-minute goal ended any hope that Inter Miami had to complete a stunning playoff charge, and the defeat coupled with CF Montreal’s big 4-1 victory over Portland means Inter Miami are officially out of the postseason running.
  Fact 1.0: Alvaro Barreal scored a goal in the 78th minute.
  Fact 1.1: Barreal's 78th-minute goal ended any hope Inter Miami had of completing a stunning playoff charge.
  Fact 1.2: Inter Miami's defeat, together with CF Montreal's 4-1 victory over Portland, means Inter Miami are officially out of the postseason running.
Sentence 2: Lionel Messi's arrival at Inter Miami has brought with it a craze for fans to get a glimpse of one of the sport's most talented and legendary players of all time.
  Fact 2.0: Lionel Messi's arrival at Inter Miami has brought a craze for fans to get a glimpse of him.
  Fact 2.1: Lionel Messi is one of the sport's most talented players of all time.
  Fact 2.2: Lionel Messi is one of the sport's most legendary players of all time.

Retrieved items:

=== END ITEM id=9483edeee994808f ===
