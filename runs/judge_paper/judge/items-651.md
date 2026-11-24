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

=== ITEM id=71c177c8bdedcf9a ===
Gold sentences and facts:
Sentence 0: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.0: Epic chose to focus on simply painting Google as the bad guy on day one.
  Fact 0.1: Epic's focus on painting Google as the bad guy happened on day one of the trial.
  Fact 0.2: The issue of painting Google as the bad guy will probably come up later in the trial.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: Epic will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 1>

<item 2>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 2>

<item 3>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 3>

<item 4>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 5>

<item 6>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 6>

<item 7>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 7>

<item 8>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 8>

<item 9>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 9>

<item 10>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 10>

<item 11>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 11>

<item 12>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 12>

<item 13>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 13>

<item 14>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 14>

<item 15>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 15>

<item 16>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 16>

<item 17>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 17>

<item 18>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 18>

<item 19>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 19>

<item 20>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 20>
=== END ITEM id=71c177c8bdedcf9a ===

=== ITEM id=3b1320d9a074de7e ===
Gold sentences and facts:
Sentence 0: Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right things.
  Fact 0.0: Erik ten Hag is a top coach from Ajax.
  Fact 0.1: The party referred to as they needs a top blueprint not only for now but for the next five years.
  Fact 0.2: The party referred to as they needs to work towards that blueprint.
  Fact 0.3: On the journey toward that blueprint, there are sometimes hard moments when the party does not get the right results.
  Fact 0.4: The hard moments happen even though the party is doing the right things.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United's exit followed a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.3: Kingsley Coman's goal was the one that sealed the 1-0 win.
  Fact 1.4: The 1-0 win was for Thomas Tuchel's side.

Retrieved items:
<item 1>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
We should do this on a consistent basis.

"We don't take the benefit in the attacking transition moments, we should be because the ability in the team can take benefit from it. We pay attention to it,
we need to grow. Then games will become easier."

So, what do the stats say?

Sky Sports' Sam Blitz:

Ten Hag is indeed right - Manchester United do rank very highly in their pressing stats in the final two-thirds of the pitch.

United are, as he says, "top of the Premier League" in terms of possessions won in the final third and high turnovers. He is slightly wrong in terms of winning
the ball in the middle third of the pitch - with United second behind Everton in that statistical table.

However, there is still work to be done in terms of improving United's pressing. Their passing permitted per defensive actions (PPDA) tally ranks 10th - so,
average - in the league.
</item 1>

<item 2>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
The match on Wednesday is not a knockout game but in truth the scenario is not much different. Defeat against Galatasaray would again end United’s Champions
League’s campaign prematurely while a win, assuming Copenhagen lose against Bayern Munich, would put qualifying for the latter stages back in their hands.
United will face the same atmosphere as they did 30 years ago because the home fans know victory would turn their subsequent trip to Denmark into a potential
shootout for who goes through with Bayern, with the achievement of knocking out a club with the storied history of United only adding to the pleasure.

In 1993 everyone was on a learning curve at United. It was their first return to Europe’s top table after an absence of 24 years. Much of the squad had won the
European Cup Winners’ Cup in 1990‑91 but this was a clear step up.
</item 2>

<item 3>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Some inside the club believe Murtough will either be moved sideways or follow Arnold through the exit door. Ratcliffe might have signalled his assessment of him
when, during a March tour of the club, he questioned the signing of a 30-year-old Casemiro on a four‑year deal worth about £350,000 a week. Murtough, whose
football department has a transfer veto, and Ten Hag, who has the same, were responsible for the Brazilian’s signing. The word is that Murtough, present when
Ratcliffe raised this, was hardly impressed. Particularly as Nice, the French club owned by Ratcliffe, recruited Aaron Ramsey, Kasper Schmeichel and Ross
Barkley as part of the nine signings in the same summer Casemiro joined United. Schmeichel was 35 and signed a three-year contract in a £1m transfer from
Leicester; Ramsey, then 31, and Barkley, then 28, came on frees and were given one-year deals.
</item 3>

<item 4>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
The 1990s were a hell of a decade. They still are around here.

'We've lost decisive players': Ten Hag says injuries were key to Champions League exit – video

So much for the good news then. The bad news is that Ten Hag’s United have now compiled arguably the worst performance by any English team in a Champions League
group stage. OK, it’s no Blackburn Rovers. But there are extra marks here for United’s pedigree, for the fact the group really wasn’t much of a group, and for
United managing to finish bottom, shipping 15 goals in the process, more than any other English team ever.

Plus of course there were just so many moments of farce along the way, a six-part saga of chaos, collapse and laughter in the dark. We remember Onana diving out
of the way of the ball in Munich. We remember Onana also diving out of the way of the ball in Istanbul.
</item 4>

<item 5>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
They're easy saves and he's cost his team.

"If Onana doesn't make those two mistakes, I think United go on to win 4-1.

"He's so laid back on the ball that he can't switch on, he's late to dive for the free-kicks."

4 United now face a huge task in their hope of reaching the last 16 Credit: AFP

United's 12 goals in the group stage is the fifth highest tally out of any club but their defensive woes have been their downfall, and former defender Rio
Ferdinand cites a lack of balance as to why.

The 2007/08 Champions League winner with Man United said: "I think they've got to look at themselves.

"You can't go into a Champions League campaign and score nine goals in three away games and have one point at the end of it. I think that shows you that there's
a lack of balance in the team at the moment.
</item 5>

<item 6>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
As Manchester United's season continues to slip away, the pressure around Dutch boss Erik ten Hag continues to get more palpable.

The latest slip saw the Red Devils crash out of Europe as they fell 1-0 to Bayern Munich on the final day of Champions League group-stage play, finishing bottom
of their four-team group to leave them eliminated from Europe in their worst performance in the competition since 2005.

On top of that, they set the record for most goals against in a Champions League group stage by an English club, with 15 conceded in their six games. Their
defensive record has collapsed as the season goes on: from a Premier League standpoint, they sit eighth in the English top flight, but their -5 goal
differential is the worst of any club in the top 10.

With Jim Ratcliffe having now completed the formalities around his investment into the club, taking control of 25% of his boyhood club after a year of
negotiations, the pressure will be piled on even more.
</item 6>

<item 7>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
“Sometimes players need time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen but there was this narrative
that started to grow internally and externally that he wasn’t a good finisher.

6 Mitchell is becoming an increasingly recognisable face

“I remember telling a coach ‘you’re wrong, his history shows in two different clubs he’s a very astute finisher off both sides’.”

Mitchell remains good friends with Rangnick from their time together at Leipzig, and last season made headlines when he was spotted at a youth team game while
the German was interim coach at Man United.

So astute is Mitchell in his observations, he even laid out a blueprint for United's success while at Monaco.

“I think context is important to measure what Manchester United's current situation is, but I think they do need to get to the point of really putting down what
Manchester United needs to be in the modern game,” he explained.

"We are a long time from the Sir Alex Ferguson era.
</item 7>

<item 8>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 8>

<item 9>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
The manager’s future

Erik ten Hag’s team are treading water at precisely the wrong moment: as Sir Jim Ratcliffe takes charge of Manchester United’s sporting department before making
sweeping changes in the posts that will directly affect the manager. Richard Arnold has left despite officially staying until the end of the year and into his
role may come Jean-Claude Blanc, the CEO of Ratcliffe’s Ineos Sport, as the man Ten Hag reports to. Ten Hag’s line manager may change too, as Ratcliffe
considers whether a new football director is required, a position held by John Murtough. Ten Hag may have a first clue of his status in Ratcliffe’s new empire
if he is sounded out for his view of a prime factor in his ability to coach and manage the side to success: United’s transfer policy. This leads us to Murtough
…

Will Murtough survive?

The football director has no plans to depart but whether he remains in situ may be a matter for Ratcliffe.
</item 9>

<item 10>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
They still have the FA Cup, and the extended pursuit of fourth or even fifth place, although the state of the coefficient means the thing that might keep
Manchester United out of Europe next season could be Manchester United being so bad in Europe this season. Maybe City can revive their fortunes by winning the
Champions League again.

Ten Hag will take the blame of course for the early exit and plenty of mistakes have been made, not least in the players the manager has been allowed to sign.
In reality assessing the job any United manager does with this team involves applying a series of filters. To be United manager is to undertake three concurrent
roles.

First you must manage the past, which is constantly in the room and which skews every act, achievement and expectation. Second you must manage the dysfunction
of the present, which expresses itself in every detail from a leaking stadium roof to the deep ills of a playing squad peopled with squatters and long‑term
missteps.
</item 10>

<item 11>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Almost certain to consult Rangnick before any approach from Ratcliffe, Mitchell may be warned off of a club where long-term thinking hasn’t existed.

But, should Ratcliffe get his man, it could signal exactly the change in fortunes he and every supporter has been hoping for at Manchester United.
</item 11>

<item 12>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
"So I think there's a lot of question marks in the performances and the way they're going about things at the moment but balance for me is one of the key
aspects that they've got to really find.

"We mentioned in commentary, scars from throwing away leads, those are hard to get away, you've got to really manage games and he mentioned that himself Bruno,
you've got to go out managing games correctly.

"They've got to do that not over one or two or three games, they've got to do that over a period of games now so they can erase those scars that they've created
and they'e grown through bad results and throwing away leads, especially in Europe."
</item 12>

<item 13>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
That raises doubts over the future of current sporting director John Murtough and chief executive Richard Arnold, although manager Erik ten Hag is believed to
be on a solid footing despite United’s poor start to the season.

Expect Sir Dave Brailsford, the mastermind behind Team GB’s cycling success at the London 2012 Olympics and director of sport for Ratcliffe’s company INEOS, to
have a prominent role in any reshuffle.

Ratcliffe also has ambitious plans to upgrade Old Trafford and turn it into a 90,000-seat stadium.

He wants the deal ratified in time for the January transfer window, however due to FFP concerns, there is no guarantees United will be able to spend big.

Despite the fact he turns 71 a day before the board meeting, it appears Ratcliffe’s plans centre around long-term stability and success rather than a short-term
turnaround in United’s fortunes.
</item 13>

<item 14>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
After all that pressure, all that tender hope, it looked a bit like a training goal.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

And that was pretty much that. From here United’s season continues to narrow. They can’t win the league now, can’t win the Champions League, can’t win the
Carabao Cup.
</item 14>

<item 15>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
In the second half United’s back five was André Onana, Aaron Wan-Bissaka, Jonny Evans, Raphaël Varane and Diogo Dalot, an entirely random collection of players
signed, at various times, by Ten Hag, José Mourinho, Ole Gunnar Solskjær and Alex Ferguson. Presumably Daley Thompson, Michael Portillo and Howard from Take
That were busy.

By the end United had five academy players on the pitch, ages ranged between 17 and 35, in outline evidence of internal good health, in reality another symptom
of a machine that wants to function properly but has for so long been made to judder along at half-speed. Ever feel like you’re starting to slip into the past
tense for good? When VAR happens at Old Trafford there isn’t even a VAR screen to say “VAR Check”. The 1990s were a hell of a decade. They still are around
here.
</item 15>

<item 16>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Joel and Avram Glazer put the Red Devils up for sale in November 2022 Credit: AFP

5 United supporters are desperate to see the Glazers depart Old Trafford Credit: Twitter

The fact the Glazers will remain in situ has not gone down well with supporters, but this is seen as the first step to Ratcliffe eventually removing the
Americans and taking overall control.

The exact timescale as to when that happens remains vague and how much it will cost Ratcliffe overall could depend on United’s sporting and commercial success
in the coming seasons.

Thursday’s board meeting was pre-arranged and while it is expected that Ratcliffe’s proposal will discussed in detail, sources close to United have indicated to
talkSPORT that it is far from guaranteed it will be put to a vote with several details still to be thrashed out.
</item 16>

<item 17>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Photograph: Adam Vaughan/EPA

They held the ball nicely, probed in neat, clipped, mannered attacks. A lot seemed to hinge on how desperately Bayern felt the need to twist that scalpel,
whether they could really be bothered to take this thing to the wall. United defended well. Their waves of pressing drew waves of applause. Everyone here seemed
to be trying.

But with 58 minutes gone Copenhagen scored and United’s hopes began to slip beneath the waves. Just over 10 minutes later Bayern made it 1-0 in Manchester.
Kingsley Coman got it, granted a huge expanse of time and space in front of the United goal, enough to set himself, as though taking a penalty, and belt the
ball into the corner. The moment was made by a delightfully sneaky round‑the‑corner pass from Kane. After all that pressure, all that tender hope, it looked a
bit like a training goal.
</item 17>

<item 18>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
United now have to beat Bayern Munich and hope Galatasaray draw with FC Copenhagen on the final matchday to progress, and it's the third time in the Champions
League this season that they have been in a winning position and then failed to emerge victorious.

Eight points have been dropped when they've been in the lead, and no other team has suffered a worse fate this term.

Captain Fernandes appears to be fed up of their inability to hold onto results as he told TNT Sports: "Every time we concede a goal we had many chances to score
again.

"I think at the end of the game we had like four chances to score a goal but we have to get our heads up and understand that sometimes it's about finding
another pass, finding the player that is free.

"Trying to get the best one in the best position and take the chance ourselves, we have to look for the team first and after, for ourselves."
</item 18>

<item 19>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Finally you get to manage the everyday metrics on which you will be judged; team, results, style, energy, messaging. Win things, improve players, create a
coherent and happy internal culture: but do it during an unceasing takeover circus while star players sulk and brief against you, while the ghost of Hamlet’s
father still skulks in the eaves. This felt like the end of one thing, perhaps even the beginning of the end of something else.
</item 19>

<item 20>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
"We are a long time from the Sir Alex Ferguson era. Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five
years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right
things.

"You have to know that, over that period, the consistency of the decision-making will yield a good end result. I think we've seen that both Manchester City and
Liverpool, that when you adopt that long-term strategy you can get real consistency and sustainable success.

"I think that's the biggest thing for Manchester United; putting down their identity of what they want to be today, but most crucially what they want to be in
five years' time, and where they want to be, and what they want to look like; from everything from young players, to the style of play, to the whole culture
peace in Manchester United."
</item 20>

<item 21>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.
</item 21>

<item 22>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
In 2020‑21, they lost their final two group games and were eliminated, while back in 2015-16, Wolfsburg won the deciding group match to knock out United.

Mauro Icardi celebrates scoring the winner for Galatasaray in their 3-2 win at Old Trafford in early October. Photograph: Darren Staples/AFP/Getty Images

When it comes to pressure in the Champions League of late, United do not cope well and there are few events trickier than a night in Istanbul. Pallister smiled
at a bell boy at the team hotel and got a finger across the throat in response, Brian McClair had a cleaner bang on his bedroom ceiling in the night and other
players received numerous phone calls to disrupt their sleep. The current squad could face the same level of malevolence and need to be prepared.

United have done little learning in recent times, repeating the same failures in Europe’s elite competition.
</item 22>

<item 23>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 23>

<item 24>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Manchester United’s century old stadium needs an upgrade. Photograph: Greig Cowie/Shutterstock

Communications

The Glazers treat a microphone or TV camera as United fans do Liverpool or Manchester City: as a fierce enemy. No family member has given a press conference or
interview during their 18-year tenure.
</item 24>

<item 25>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
As the seconds ticked down at the end of a Champions League season‑ender that felt like a gentle, even quite tender, act of sporting euthanasia, the only noise
inside Old Trafford was the sound of the Bayern Munich fans singing an impressively sustained version of Football’s Coming Home. A little later they sang Is
This A Library? Actually, no, it’s more of a museum. They chanted “Auf Wiedersehen” at the departing members of the home support who, frankly, had done pretty
well to stick it out to that point.

The good news for Erik ten Hag at the end of this low-fi 1-0 defeat is that United’s players didn’t lie down and die at Old Trafford. Instead they stood up and
died, running hard and creating a kind of simulacrum of a functioning elite team.

This was not a group of players in revolt or drained of spirit. They looked like what they are, a muddled creation, stitched together out of parts and off-cuts.
</item 25>

<item 26>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
United have done little learning in recent times, repeating the same failures in Europe’s elite competition. It would not be a disgrace to lose against a good
Galatasaray team, especially given Erik ten Hag is missing key players, but if the worst does happen, everyone will need to come out of the experience stronger.
Regardless of the result, it should be a night of education when it comes to the demands of the Champions League.

Six of United’s squad that failed in 1993 went on to win the competition six years later. “It did galvanise us,” says Mike Phelan, who started in central
midfield 30 years ago. “Winning the Cup Winners’ Cup took us to another level and we felt pretty comfortable with each other and felt we could develop into a
serious team, and I think going into the Champions League was one of those moments where we expected to do well – the experiences along the way were great and
took us to the next level further down the line.
</item 26>

<item 27>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
No family member has given a press conference or interview during their 18-year tenure. Ratcliffe has arrived on a ticket of, partly, being the local lad made
good (Failsworth, where he was born, is next to Newton Heath, United’s birthplace) so if he proves as silent as the Americans then this will be: a) poor PR; and
b) just plain poor. Football fans can be classed as the most loyal citizens of any society but even the faith of United enthusiasts has been stretched by the
cold relationship created by the Glazers’ stance regarding communication. A tap-in, then, for Ratcliffe is to remedy this by opening and maintaining a regular
line into supporters.

The Glazer family have little relationship with the Manchester United fans – something Ratcliffe would do well to avoid.
</item 27>

<item 28>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Old Trafford (and the training ground)

To find an apt shorthand for the lurching ship that is United, end the search at the roof of their stadium where on Manchester’s (many) wet days rain leaks
through. Old Trafford remains a storied venue but as it continues to crumble the tale of disrepair has become a central narrative in its recent history. As
Premier League titles have become scarce (zero in a decade) the tale of neglect of a ground that dates to 1910 is emblematic of the Glazers’ quasi‑absentee
proprietorship. Ratcliffe will invest $300m (£237m) into infrastructure but this is small beer when considering the multibillion-pound facelift or new venue
required. There is also the cramped Carrington base: as on Sir Matt Busby Way, the footprint is large at the training ground but the configuration and
facilities are ageing – badly.
</item 28>

<item 29>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Only leaving the side to reach the pinnacle of his job, finally stepping up from a head of recruitment to become sporting director at Monaco, Mitchell showed
just how big picture he is with an interview in 2021.

“I want to continue to drive standards in each and every department,” he said. “Psychology, sports science, nutrition, medicine, the way we travel, the way we
refuel.

“Everything, every element I think we need to keep pushing to its maximum to try and be the best we possibly can.”

Having worked in football management for over a decade, Mitchell’s career has always been about the long term for his club’s, with the young signings he’s made
often only paying off after he’s left.

One in particular was 2022/23 Premier League top scorer Son, who perfectly sums up his philosophy.

“I took quite a lot of criticism in my first year at Tottenham for some of the decisions and Sonny was kind of at the centre of that,” he explained.
</item 29>

<item 30>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Photograph: Michael Regan/Getty Images

Ensuring a working relationship with the Glazers

For paying an overpriced £1.3bn for his 25% Ratcliffe obtains a major element of control but still the Glazers remain majority owners. The 71-year-old oversees
football policy but these do not exist in a vacuum given that the commercial side affects this and vice versa. Ratcliffe accepts Manchester United have an
impressive revenue-based model that makes the club akin to a cash machine but he is not enamoured of how the money generated is invested – primarily in the
squad. What happens then, for example, if Ten Hag wants a new midfield ace, Ratcliffe agrees but the Glazers cite a downturn in income and thus seek to block
the signing?
</item 30>
=== END ITEM id=3b1320d9a074de7e ===

=== ITEM id=34d6a8444f4c1c0e ===
Gold sentences and facts:
Sentence 0: McTominay is now United's top scorer for the season with six goals, one ahead of Bruno Fernandes and Rasmus Hojlund.
  Fact 0.0: McTominay is now United's top scorer for the season.
  Fact 0.1: McTominay has six goals for the season.
  Fact 0.2: McTominay is one goal ahead of Bruno Fernandes.
  Fact 0.3: McTominay is one goal ahead of Rasmus Hojlund.
Sentence 1: He has already been firing in many top-flight goals this term - as well as being the joint-leading scorer from the Champions League group stages.
  Fact 1.0: McTominay has already been firing in many top-flight goals this term.
  Fact 1.1: McTominay is also the joint-leading scorer from the Champions League group stages.

Retrieved items:
<item 1>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
Santiago Gimenez (Feyenoord/Mexico) - 39 goals

One of the most exciting strikers in world football, the 22-year-old has taken European football by storm.

Joining Dutch side Feyenoord in the summer of 2022, he fired Arne Slot's men to the Eredivisie championship with 15 goals.

With 20 goals this season along with two strikes for Mexico, it's clear to see why Arsenal, Tottenham and Chelsea have all been linked with a move.

8 Gimenez seems destined for a big move in the near future Credit: Getty

5. Romelu Lukaku (Roma/Belgium) - 40 goals

After reuniting with Jose Mourinho at Roma in the summer, Lukaku has found his goalscoring form once again.

The Belgian has been firing back in Serie A following another loan move from Chelsea, netting in almost all of his starts this season.
</item 1>

<item 2>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

On another night, perhaps Chelsea score from one of these counterattacks. Perhaps in another game United come up against a better finisher than Nicolas Jackson,
who for all his talent looks like a man playing football in a hall of mirrors. This is the high-wire act that Ten Hag is enacting: a daring gamble that United
will gain more going forward than they lose going back.

That once Lisandro Martínez and Raphaël Varane come back, once he brings the defensive line up 10 yards, once Rasmus Højlund hits some form, once Mason Mount
gets up to speed, lawless abandon will begin to resemble a sustainable attacking blueprint.
</item 2>

<item 3>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
32nd minute: McTominay should have his second! Fernandes picks him out beautifully from the left but the Scotsman heads directly at Sanchez. Either side of the
Chelsea goalkeeper and United would have a two-goal lead.

25th minute: Penalty appeal for Chelsea! Sterling drops to the floor after very minimal contact with Victor Lindelof but neither the referee nor VAR buy his
theatrics.

19th minute: GOAL! 1-0 United! Scott McTominay finishes off a scrappy period of play and United have a deserved lead.

18th minute: Silva's block denies Antony the opener. He's not scored at Old Trafford since his debut...

15th minute: It's all gone a bit wild here as neither side appears to have a midfield capable of dictating the play. Chelsea have just had two glorious chances
but Mudryk couldn't beat Onana and Fernandes made a brilliant recovery run to prevent the other attack.

9th minute: SAVED!
</item 3>

<item 4>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
United fans will no doubt note the rich irony in the fact that these days McTominay is basically doing the job they once paid Paul Pogba to do. Marauding
midfield engine and late-running goal threat; the all-action superhero who can chip in with 10 goals a season and pull off slide tackles that make the highlight
reels. It never quite worked out for Pogba at United, largely – if you believe his most vehement critics – on account of his hair. McTominay is a different kind
of player with a different kind of skillset. But for all the undoubted leadership qualities of Bruno Fernandes and Harry Maguire, tactically it is McTominay
running the show right now: a development with numerous interesting consequences.

Here he scored his fifth and sixth goals of the season. Previously he has never got more than seven. Where is this coming from?
</item 4>

<item 5>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Imagine what Manchester United could do if Erik ten Hag hadn’t already lost the dressing room. Yes, it’s been another of those weeks at Old Trafford: rancour,
rumour and recrimination, barbs in the press and barbs at the press, defeat at Newcastle followed by this stirring recovery against Chelsea, and the startling
realisation that United are now three points behind Manchester City. It’s still only Thursday, by the way.

Perhaps ultimately this win only buys Ten Hag a few days’ grace, a warm flume of goodwill that lasts only as long as it takes for Dominic Solanke to run through
on goal for Bournemouth on Saturday afternoon. Back in the grip of crisis, back in the now-familiar lexicon of surly unsourced stories about dressing room
discontent. Such is the way of things at football’s most reliable content provider. This beast must always eat.

But block out the noise for just a moment, as of course United cannot, and something interesting is happening here.
</item 5>

<item 6>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
His latest acrobatic effort is well wide of the goal.

49th minute: Jackson has a glorious chance at the back post to make it 2-1 but he can't keep his header down!

46th minute: Both sides make a change at the break. Sergio Reguilon comes on for Lindelof and much to the delight of the Chelsea fans, Reece James comes on for
Cucurella.

HT: That's all from a hectic first half, hopefully more of it to come in the second!

45th minute: GOAL! 1-1! The former City man finds the bottom corner with a delicate stroke past Onana. It's a wonderful goal from Cole Palmer to pull level
right before the break.

36th minute: And that should be the equaliser! Mudryk finds himself in the box on his weak foot but the Chelsea forward drags his effort wide.

32nd minute: McTominay should have his second!
</item 6>

<item 7>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
Kane and Ronaldo, two national-team record scorers still operating in 2023 (Burak Akbulut/Anadolu Agency/Getty Images)

Considering how many hat-tricks Haaland scores for Manchester City, it’s not beyond the realms of possibility that he could score six goals during this
international break to move level with Juve, particularly given Norway’s first fixture is a home friendly against the Faroe Islands today (Thursday), before a
European Championship qualifier against Scotland in Glasgow on Sunday. That said, the Faroes’ defence is less leaky than you might expect — only twice in their
last 22 outings have they conceded more than three times in a game.

Advertisement

The wider question is whether we will ever see Haaland at a major tournament.

Despite the presence of him and Arsenal captain Martin Odegaard, Norway have failed to qualify from their Euro 2024 qualification group directly, with Spain and
Scotland already securing the top two spots.
</item 7>

<item 8>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Which, as the German champions have not lost in 40 group-stage games in this competition and United have not beaten anyone resembling a top team this season,
always felt unlikely.

This was far from their worst performance of the season, nowhere near as bad as Bournemouth or Crystal Palace or Newcastle, or the other defeat to Newcastle.
They were not as self-destructive as they had been in Denmark and Turkey and at home to Galatasaray. But they still lost.

Harry Maguire’s injury forced him off the pitch as 36-year-old Jonny Evans replaced him (EPA)

And they still went out, with four points, in fourth place. Go back a couple of months and Raphael Varane had said United could win the Champions League. On a
night when Ten Hag seemed to revise his opinion of the quadruple Champions League winner, the defender may need a rethink. Instead, United won a lone game in it
and even that required a penalty save to deny FC Copenhagen a draw.
</item 8>

<item 9>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
"I had to set some standards"



Erik ten Hag claims there was not a good culture among staff and players before he joined Man United 😬pic.twitter.com/DJWe42mDXY — Sky Sports Premier League
(@SkySportsPL) December 5, 2023

Man United vs Chelsea lineups

Midfielder Casemiro and defender Lisandro Martinez remain out for Manchester United, while Christian Eriksen may not return in 2023.

Mason Mount missed United's defeat at Newcastle United on Saturday and Jonny Evans has not played for almost a month because of a thigh issue. Amad Diallo and
Tyrell Malacia have long-term injuries.

Man United lineup (4-2-3-1, right to left): Onana (GK) — Dalot, Maguire, Lindelof, Shaw — Amrabat, McTominay — Antony, Bruno Fernandes, Garnacho — Hojlund.
</item 9>

<item 10>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
"We started very good, proactive, brave, we created a lot of chances in that moment, we could have gone up with two or three goals."

Man United vs Chelsea final score

Final score Goalscorers Man United 2 McTominay 19', 70' Chelsea 1 Palmer 45'

Man United vs Chelsea live updates, highlights from Premier League

FT: United can breathe a sigh of relief and hold onto a crucial three points!

90th minute: Broja hits the post! That was so close to the equaliser for Chelsea. There are four minutes of added time.

80th minute: Garnacho so nearly makes it 3-1 to United but his delicately finish toys with the goalline.

74th minute: An insane recovery run and block by Reece James denies Garnacho from doubling United's lead. He was picked out superbly by Onana to kickstart a
rapid counter by the hosts.

70th minute: GOAL! 2-1 United!
</item 10>

<item 11>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
The Portugal international Ronaldo has been battling with with Kylian Mbappe, Erling Haaland and Harry Kane for the title of the best goalscorer of 2023..

Here talkSPORT.com runs you through how Ronaldo compares to the top six highest scoring European based footballers this year...

7. Lautaro Martinez (Inter Milan) - 37 goals

All of Martinez's goals this year have come at club level for Inter Milan, with none coming on the international scene for Argentina.

With so many goals this campaign, it's no surprise that the Serie A giants are desperate to tie him down to a new deal.

He currently leads the Scudetto Golden Boot race - six more than closest rival Domenico Berardi.

8 Inter striker Martinez has carried on from last season by banging them in for fun Credit: Getty

6.
</item 11>

<item 12>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Varane was summoned from his inexplicable exile, after a 10-game absence when both Luke Shaw and Jonny Evans were preferred at centre-back to a World Cup
winner. He performed with an assurance to make his omission all the odder, but it has scarcely been the only U-turn in selection; with Harry Maguire and Scott
McTominay, Ten Hag is compiling a group who were first out of his plans and then back in. For a manager who likes to project an image of strong, decisiveness
leadership, he has had to reverse course rather too often. But then his team are going backwards.

When Coman scored, United became the first Premier League side to concede 15 goals in a Champions League group stage. When Maguire hobbled off, holding his
groin, and Evans came on, United had a man who played in United’s 2007-08 Champions League-winning campaign, dragooned into service 16 years later.
</item 12>

<item 13>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
Haaland is currently averaging nearly a goal a game for his country, which will inevitably be difficult for the 23-year-old to sustain over his career. But it’s
worth pointing out how impressive that is, even at this early stage. Again, excluding countries who have never qualified for a World Cup, only the
aforementioned quartet of Juve, Riva, Puskas and Nielsen, plus Japan’s Kunishige Kamamoto, hold their nation’s international goalscoring records and also boast
a rate of 0.75 goals per game or more.

Even Cristiano Ronaldo, the most prolific international goalscorer of all time with 127 for Portugal, boasts ‘only’ 0.63 goals per game, a lower rate than the
likes of Romelu Lukaku (Belgium), Kane and Aleksandar Mitrovic (Serbia), which owes to his early days as a winger rather than a central striker.
</item 13>

<item 14>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
This was a win that felt a lot tighter than it really was: United inexplicably hanging on at the end despite enjoying 28 shots and an xG tally of 4.07. For long
periods United played some of their most authentically thrilling football of the season, a doctrine of relentless attack that had Diogo Dalot flying forward
from right-back even as the minutes ticked down.

It was also a game that felt more chaotic than it really was. In fact the slippery, slingy nature of the play was largely by design: United hollowing out their
midfield in order to throw bodies forward, Chelsea hollowing out theirs in order to leave men up for the breakaway. It was a game that hung on United’s
determination to keep allowing Chelsea counterattacks, and Chelsea’s determination to keep messing them up. It was a game, above all, that hung on the curious
main character energy of Scott McTominay.
</item 14>

<item 15>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
He has also found his shooting boots for his country, most notably scoring four in a 5-0 win over Azerbaijan in November.

Lukaku's career seems to have reignited and that has seen him close to claiming a place as one of the highest goalscorers this year.

8 Lukaku ended the Euro 2024 qualifying campaign as top scorer Credit: AFP

4. Erling Haaland (Manchester City/Norway) - 50 goals

No player in Premier League history has made more of an impact following their arrival than Haaland, who broke the goal record for a single season with 36
strikes last term.

He has already been firing in many top-flight goals this term - as well as being the joint-leading scorer from the Champions League group stages.

Just six out of his 50 goals in 2023 have been for his national side Norway, with Haaland helping the Citizens to a historic treble last campaign.
</item 15>

<item 16>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.
</item 16>

<item 17>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
They are, at least, likely to qualify for the play-offs, and therefore will have two must-win games in March to secure their first major tournament appearance
since 2000 — the summer when Haaland was born. But there’s been little in recent performances to think Norway will breeze through those play-offs.

Juve’s individual record will soon be surpassed, but captaining his side to a bronze medal at the Olympics may stand as his nation’s greatest achievement for
much longer.

GO DEEPER Erling Haaland is phenomenal - so why hasn’t he made Manchester City better?

(Top photo: Sebastian Widmann – UEFA/UEFA via Getty Images)
</item 17>

<item 18>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Man United subs (12): To be confirmed

Ben Chilwell remains unavailable for Chelsea and Malo Gusto has been missing with a knee injury, but fellow defender Reece James is available again after
suspension, as is Marc Cucurella.

Romeo Lavia and Christopher Nkunku are likely to still be sidelined and Conor Gallagher is banned after receiving two yellow cards against Brighton. Carney
Chukwuemeka, Trevoh Chalobah and Wesley Fofana are out.

Chelsea lineup (4-3-3, right to left): Sanchez (GK) — Disasi, Silva, Colwill, Cucurella — Sterling, Fernandez, Caicedo — Mudryk, Palmer, Jackson.
</item 18>

<item 19>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Previously he has never got more than seven. Where is this coming from? Partly from the fact that opponents have still not quite cottoned on to McTominay’s goal
threat, those late runs to the far post, the blur on the edge of the 18-yard box that you don’t spot until it’s too late. But partly, too, it is a function of
how Ten Hag is using him.

Erik ten Hag (right) and Mitchell van der Gaag, his assistant, talk tactics during the game. Photograph: Ash Donelon/Manchester United/Getty Images

Unlike his previous more conventional midfield roles, these days McTominay basically has a licence to create overloads wherever he finds them. Against low
blocks and structured defences – and nobody is remotely accusing Chelsea of possessing one of those – he provides the extra body in a tightly marked penalty
area.
</item 19>

<item 20>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
"In high ball regains, we are a top of the Premier League. In the middle [of the pitch] ball regains, we are top. The pressing is very good. So that's not the
truth, what he [Carragher] is telling."

Please use Chrome browser for a more accessible video player FREE TO WATCH: Highlights from Sheffield United’s Premier League clash with Manchester United

In a fortnight where United have picked up three straight wins against Brentford, Sheffield United and FC Copenhagen, albeit via far from convincing
performances, Christian Eriksen said United still need to find their "balance and routines" this season.

Asked about those comments, Ten Hag said: "We are in the right direction. We have a way to go. I see positives but I see negatives.

"What Christian is referring to is we are not consistently in possession on the level where we can be. Like Arsenal away, the first 30 minutes in Bayern Munich
away. We should do this on a consistent basis.
</item 20>

<item 21>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
8 Kane already has 25 goals for Bayern in all competitions Credit: Getty

1. Cristiano Ronaldo (Al Nassr/Portugal) - 53 goals

Ronaldo has chipped in with 43 goals for Al Nassr to prove he never left his shooting boots back in Europe.

He has even banged in ten times for Portugal, as he helped the Euro 2016 winners triumph in all ten of their Euro 2024 qualifying matches.

His half-century in 2023 is the eighth time in his career that he has scored at least 50 goals in a calendar year.

His two strikes in the 5-2 win against Karim Benzema's Al Ittihad saw him leapfrog above Kane and Mbappe.

And with Al Nassr still having one match left before 2024, he could add to his ever-growing tally.
</item 21>

<item 22>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
United now have to beat Bayern Munich and hope Galatasaray draw with FC Copenhagen on the final matchday to progress, and it's the third time in the Champions
League this season that they have been in a winning position and then failed to emerge victorious.

Eight points have been dropped when they've been in the lead, and no other team has suffered a worse fate this term.

Captain Fernandes appears to be fed up of their inability to hold onto results as he told TNT Sports: "Every time we concede a goal we had many chances to score
again.

"I think at the end of the game we had like four chances to score a goal but we have to get our heads up and understand that sometimes it's about finding
another pass, finding the player that is free.

"Trying to get the best one in the best position and take the chance ourselves, we have to look for the team first and after, for ourselves."
</item 22>

<item 23>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
United have now won their past 19 Premier League games at Old Trafford when scoring first, including all 16 under their Dutch manager.

MORE: Every game on Amazon in December and how to watch

Ten Hag: Crisis? What crisis?

The United manager faced a demanding week amid reports indicating disunity within his team and his players did exceptionally well to block out the noise and
focus on footballing matters.

When asked about a potential crisis, he said: “Crisis? Not for us. We keep calm. We know where we have to build. We are going in the right direction. This team
is capable to be successful.”

And United were successful tonight, thanks to McTominay. Ten Hag explained how his midfielder has managed to become more involved in an attacking sense this
season: "He has a very good smell when to arrive and a very good finish.

"It is not a frustration, in the end, we won but we could have made our lives much easier," he said.
</item 23>

<item 24>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
9th minute: SAVED! It's a woeful penalty from Fernandes and Sanchez makes an easy diving save to his right.

7th minute: PENALTY! It's United who are rewarded for their bright start as Antony was brought down by Enzo Fernandez in the box. Chris Kavanagh originally
waved play on, however, VAR instructed him to take a look at the monitor and the referee gives the penalty.

4th minute: Sanchez makes a great save to deny Hojlund the opener, who was well found by Luke Shaw.

3rd minute: Dalot picks out Fernandes on the edge of the box who blasts well over the bar with a first-time strike.

1st minute: The visitors get us underway.

15 mins until kickoff: The teams have headed down the tunnel here at the Theatre of Dreams and kickoff is next...

30 mins until kickoff: The warm-ups are underway at Old Trafford on a chilly night.
</item 24>

<item 25>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
The youngster then cut the ball back to Harry Maguire whose effort rebounded perfectly into the path of McTominay. The Scotland international took one touch
before blasting past Sanchez.

United led for the majority of the first 45 minutes but failed to build on their lead and were pegged back by the former Manchester City man, Cole Palmer. The
21-year-old, who made the transfer from the Etihad Stadium to Stamford Bridge in August, stroked the ball into the far corner with deadly accuracy. It was his
fifth Premier League goal of the season and his first from open play.

But United continued to put Chelsea under pressure in the second half and after numerous attempts, McTominay headed powerfully past Sanchez. The Scotsman had
seven shots, his most ever in a league game. Despite a flurry of late attacks driven by Reece James, Ten Hag's team stood strong and held on to see out the
victory.
</item 25>

<item 26>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
The UEFA Champions League has reached the end of its group stage, with 16 precious places in the first knockout round now claimed.

Manchester City are looking to go back-to-back after their triumph in last season's Champions League, potentially becoming the first English club since
Nottingham Forest in 1980 to successfully defend the trophy.

The likes of Bayern Munich, Real Madrid and Barcelona have also enjoyed strong group-stage campaigns and will surely be in the mix towards the sharp end of the
competition. Atletico Madrid and Real Sociedad will complete the Spanish contingent in the next round.

Manchester United and Newcastle United will play no further part in Europe, though, having been eliminated by finishing bottom of their groups. Paris
Saint-Germain will be in the Round of 16, though.

The Sporting News reviews the group stage ahead of its final match night and examines what teams from each group need to qualify for the Round of 16.
</item 26>

<item 27>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Manchester United got themselves back to winning ways thanks to a superb Scott McTominay brace against Chelsea.

The victory remarkably puts them within three points of rivals Manchester City, after the reigning champions were defeated by Aston Villa, and keeps them in
reach of the top four places.

McTominay is now United's top scorer for the season with six goals, one ahead of Bruno Fernandes and Rasmus Hojlund. The Red Devils' captain had a first-half
penalty saved by Robert Sanchez and United's Danish striker was substituted after another Premier League game without a goal, despite making 11 appearances.

Erik ten Hag's side were dominant in the first half and finally pounced on Chelsea's inability to take control in midfield. Mauricio Pochettino's men attempted
to break at pace but Sofyan Amrabat made a vital interception and Fernandes found Alejandro Garnacho out wide.
</item 27>

<item 28>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

They sang the name of Erik ten Hag’s predecessor. Not in a gesture of dissent, but as a reminder of a time when Manchester United were Europe’s best and an
evening without which a certain Norwegian would never have become their manager two decades later. After all, who put the ball in the Germans’ net? Ole Gunnar
Solskjaer. But that was a different millennium, a different United. Nostalgia underpinned Solskjaer’s management.
</item 28>

<item 29>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
As Manchester United's season continues to slip away, the pressure around Dutch boss Erik ten Hag continues to get more palpable.

The latest slip saw the Red Devils crash out of Europe as they fell 1-0 to Bayern Munich on the final day of Champions League group-stage play, finishing bottom
of their four-team group to leave them eliminated from Europe in their worst performance in the competition since 2005.

On top of that, they set the record for most goals against in a Champions League group stage by an English club, with 15 conceded in their six games. Their
defensive record has collapsed as the season goes on: from a Premier League standpoint, they sit eighth in the English top flight, but their -5 goal
differential is the worst of any club in the top 10.

With Jim Ratcliffe having now completed the formalities around his investment into the club, taking control of 25% of his boyhood club after a year of
negotiations, the pressure will be piled on even more.
</item 29>

<item 30>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Bayern Munich struck twice in the second half to come from two goals down and snatch a 2-2 draw at RB Leipzig but the champions dropped to third in the
Bundesliga with their second draw of the season.

Goals from Leroy Sané and Harry Kane cancelled out Leipzig’s first-half lead but the result left Bayern on 14 points, two behind the leaders, Bayer Leverkusen.
Leipzig, who host Manchester City in the Champions League next week, are fifth, a further point behind.

“Two very different halves from our perspective,” the Bayern coach, Thomas Tuchel, said. “In the first we started well but only for the first two minutes and
missed that huge chance.

“We were too slow, made too many mistakes and were not courageous enough. After the break we showed a different mentality. It was a bit wild, with a lot of risk
but it is a positive result, not to lose after being 2-0 down. We showed character.” Bayern travel to Copenhagen in the Champions League next week.
</item 30>

<item 31>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
The England captain is the forward United neglected to bid for in the summer. The one they did buy, Hojlund, has at least struck in the Champions League but in
domestic competitions he has been outscored 18-0 by Kane.

Yet a rookie centre-forward is not the culprit in chief. For Ten Hag, this has been a chastening setback; with Sir Jim Ratcliffe set to finalise his investment,
with United listing in the Premier League, the chances are the Dutchman has managed them in the Champions League for the last time.

Go back four years and it required a 96th minute goal by Lucas Moura to stop him from reaching the final. Now the less heralded Lukas Lerager signalled United’s
demise, putting Copenhagen ahead against Galatasaray when United needed a draw to preserve their own chances of qualification; that, and a victory over Bayern.
</item 31>

<item 32>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
We showed character.” Bayern travel to Copenhagen in the Champions League next week.

Leipzig struck twice in six minutes with Loïs Openda putting them in front in the 20th when he completed a quick break. They scored again with Castello Lukeba
stabbing in after the goalkeeper Sven Ulreich misjudged a corner.

Bayern earned a 57th-minute penalty against the run of play for a handball and Kane cut the deficit with his eighth league goal. Only Manchester City’s Erling
Haaland, when he played for Borussia Dortmund, scored more often in his first six Bundesliga games with nine goals.

The Bavarians’ comeback was complete in the 70th with Kane clearing the ball in his own half and Musiala finding Sané for the equaliser.

Bellingham celebrates scoring Real Madrid’s third goal during their 3-0 away league win over Girona.
</item 32>

<item 33>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
Erling Haaland is just six away from Norway’s goalscoring record, so it is surely inevitable that he will soon be his nation’s most prolific striker ever.

On one hand, that will have happened remarkably quickly – Haaland only scored his first goals for Norway in 2020. On the other, this has been an incredibly long
time coming.

GO DEEPER What is Haaland really like?

Of the 213 FIFA-affiliated nations, Norway’s is the longest-standing outright individual goalscoring record in the world. It was set by Jorgen Juve, a
fascinating figure who ended his international career in 1937 and later became a renowned sports journalist in his home country.

Advertisement

Juve scored a relatively modest 33 goals in 45 matches for Norway, including five hat-tricks, although his tally is made remarkable by the fact he played as a
centre-forward in less than half of those games.
</item 33>

<item 34>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Martínez then got his second goal after 77 minutes when he confidently finished a pass from Nicolò Barella with a first-time strike.

Martínez completed his hat-trick from the penalty spot with five minutes remaining after Matteo Lovato pulled Thuram’s shirt. He concluded the evening by
converting Carlos Augusto’s cross from 10 yards out.

Inter lead Serie A with 18 points, the same as their neighbours, Milan, who had briefly topped the table earlier in the day after beating Lazio 2-0 thanks to
second-half goals from Christian Pulisic and Noah Okafor. Rafael Leão played a vital role in both goals, with the 24-year-old Portugal international providing
two assists in a menacing display.

Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android
by searching for 'The Guardian'.

If you already have the Guardian app, make sure you’re on the most recent version.
</item 34>

<item 35>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
Al Nassr striker Cristiano Ronaldo looks set to be the top scorer in 2023 after his Boxing Day double against Al Ittihad.

The 38-year-old may have swapped Europe for the Middle East, but he has showed no signs of slowing down in Saudi Arabia.

8 Ronaldo has been in lethal form for club and country in 2023 Credit: Getty

Ronaldo linked up with Al Nassr in the winter transfer window after parting ways with Manchester United following his explosive interview with Piers Morgan.

Liverpool legend Jamie Carragher described the five-time Ballon d'Or winner's switch to the Gulf nation as a 'sad end' to his career, but Ronaldo continues to
prove his doubters wrong.

But he isn't the only star that has been in great form, with players across Europe banging them in for fun.
</item 35>

<item 36>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Ten Hag has just told MUTV that Chelsea do not "like to defend" and that United must take advantage of that this evening.

45 mins until kickoff: Raheem Sterling has failed to score in any of his 24 matches against Man United. Could tonight be the ex-Liverpool and Man City player's
night?

1 hour until kickoff: Team news is in! The eyebrow-raising decision from Erik ten Hag is that Marcus Rashford starts from the bench. Mauricio Pochettino starts
the former Man City man, Cole Palmer.

1 hour 30 mins until kickoff: Chelsea have won three of their last four Premier League away games but are without a league win at Old Trafford since 2013...

2 hours until kickoff: It's been a relatively busy start to the week off the pitch for Erik ten Hag, it's time to see what happens on it.

"I had to set some standards"
</item 36>

<item 37>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Goals from West Ham’s Jarrod Bowen and Mohammed Kudus extended Manchester United’s miserable run in the English Premier League.

Bowen bundled home at the second attempt in the 72nd minute, his 11th league goal this season, and Kudus struck his fourth goal in as many games to settle
matters.

It was another worrying setback for manager Erik ten Hag as United suffered a 13th defeat in 26 games in all competitions.

Facebook Twitter Whatsapp Reddit Email Share

A defensive howler from Manchester United and Mohammed Kudus capitalises! ????



What a finish from the West Ham man, but Kobbie Mainoo won't want to see this one back… ????



Stream LIVE, or on-demand after, here ????
</item 37>

<item 38>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
Ten Hag on misfiring strikers: We need to be better

Please use Chrome browser for a more accessible video player Manchester United boss Erik ten Hag speaks on the improvement he's seen in goalkeeper Andre Onana
and insists he believes Marcus Rashford can get back to scoring goals

United go into Sunday's derby with a slight striker scoring problem. City forward Erling Haaland has scored more Premier League goals this season (9) than
United's five forwards have managed in all competitions (8).

United's top scorer this season is actually defensive midfielder Casemiro with four strikes so far.

Asked if their forwards are to blame for this frontline problem, Ten Hag replied: "Fault? It's co-operation. It has to click.

"We have shown in some games, we have so many in overload positions to the opponents goal and we don't net or don't even hit the target. We should be doing this
better."

It has been a sobering start to the season for Marcus Rashford, who has one goal so far this term for United.
</item 38>

<item 39>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
Erik ten Hag has disagreed with Jamie Carragher's comments that Manchester United are going backwards this season - by saying the statistics show that his team
has actually improved.

United host local rivals Manchester City in a Super Sunday derby this weekend, live on Sky Sports with kick-off at 3.30pm, in the latest acid test to see how
far Ten Hag's side have improved from last season.

In his column in The Daily Telegraph, Carragher said United's summer signings have led to the club becoming "stale" - and that Ten Hag should now be questioned
about whether he can get the best out of his squad.

But in his pre-match press conference for the City game, Ten Hag hit back at those claims.

"In possession we have to do better, but there are reasons and I don't go into them, but everyone is seeing why. But there are also facts so I disagree [with
Carragher], we are much better," said Ten Hag.

"In high ball regains, we are a top of the Premier League.
</item 39>

<item 40>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
We should do this on a consistent basis.

"We don't take the benefit in the attacking transition moments, we should be because the ability in the team can take benefit from it. We pay attention to it,
we need to grow. Then games will become easier."

So, what do the stats say?

Sky Sports' Sam Blitz:

Ten Hag is indeed right - Manchester United do rank very highly in their pressing stats in the final two-thirds of the pitch.

United are, as he says, "top of the Premier League" in terms of possessions won in the final third and high turnovers. He is slightly wrong in terms of winning
the ball in the middle third of the pitch - with United second behind Everton in that statistical table.

However, there is still work to be done in terms of improving United's pressing. Their passing permitted per defensive actions (PPDA) tally ranks 10th - so,
average - in the league.
</item 40>

<item 41>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
There was not even the sense of heroic failure that accompanied their quarter-final exit to Bayern in 2010. There was just crushing failure. Because it wasn’t
Bayern who eliminated them. It was Galatasaray, who won at Old Trafford and scored six goals over 180 minutes against United. It was FC Copenhagen, who
qualified instead of them. Rasmus Hojlund’s former club will be in the last 16. His current one won’t. Copenhagen will be the anomalies in the draw, the
outsiders with the spot that had seemed reserved for United.

Manchester United’s defeat means they miss out on a spot in the Europa League (PA)

Instead, they are cast out of continental competition altogether. Europe’s sinking superpower have become just the fourth English team to prop up a Champions
League group when it concluded. There was something predictable about the manner of their demise; their defence was unlocked by Harry Kane, with a lovely flick.
</item 41>

<item 42>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
Kane, who has scored nine goals for England in 2023, is also the first player to register a hat-trick in their first outing against Borussia Dortmund.

It should come as no surprise that he is also on course to score the most goals in a debut Bundesliga campaign as he hunts down Uwe Seeler's record of 30 for
Hamburg in the 1963/64 season.

As well as that, Kane is on track to break Robert Lewandowski's record of 41 goals in a single season.

He's adding assists, too, as evidenced by his 'filthy' effort against Man United in the Champions League.

Like Mbappe, Kane cannot overtake Ronaldo's tally due to the German winter break.

8 Kane already has 25 goals for Bayern in all competitions Credit: Getty

1.
</item 42>

<item 43>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Stream #PLGoalRush LIVE ???? https://t.co/BkLCmmvfda#OptusSport #PL pic.twitter.com/2wvzl1wjrG — Optus Sport (@OptusSport) November 12, 2023

Only Haaland – 13 – has scored more in the Premier League this season than the Egypt international, who took his tally to 10 by scoring for the sixth successive
home game to write another entry in club’s history books.

Advertisement

Diogo Jota’s late strike then put a quick end to questions over Liverpool’s mini-stumble after the draw at Luton and Europa League defeat in Toulouse.

Aston Villa moved to fifth, three points off the top, with their 3-1 win over Fulham at Villa Park, courtsey of an own goal from Antonee Robinson before strikes
from John McGinn and Ollie Watkins.
</item 43>

<item 44>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
We remember Onana also diving out of the way of the ball in Istanbul. Casemiro’s red card, lost leads in Manchester, Marcus Rashford unfairly convicted of a
venomous stamp in Copenhagen, when frankly, he’s just not that kind of semi-interested footballer.

And yet with both group games blank at half-time United were still just one goal away here from going through to the last 16, which really would have made for
an excellent punchline. It wasn’t to be. For all their own flaws Bayern are basically a good version of whatever it is United are attempting to piece together.
Kimmich‑Goretzka‑Musiala is an A grade version of Amrabat‑McTominay‑Fernandes. Manuel Neuer is the thing Onana is trying so earnestly to be. Bayern actually did
go ahead and sign Harry Kane.

Harry Kane (centre) and his Bayern Munich teammates applaud fans after their Champions League victory at Old Trafford.
</item 44>

<item 45>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 45>

<item 46>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
70th minute: GOAL! 2-1 United! And it's McTominay who has a brace! It's a great ball to the back post by Garnacho and McTominay meets the header from inside the
six-yard box. He's in terrific goalscoring form and becomes United's top scorer this season.

69th minute: Antony tries one of his trademark curling efforts from the right after cutting in on his left but it's wide of the goal.

62nd minute: Penalty appeal for United! Shaw rockets an attempt goalwards and it looked as though Colwill may have blocked it with his hand. VAR took a quick
look and play continues.

61st minute: Antony's trivela is deflected for a corner, it may have been going wide.

54th minute: Unfortunately for Garnacho, lightning doesn't strike twice. His latest acrobatic effort is well wide of the goal.
</item 46>

<item 47>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Thiago Silva evened the score four minutes later with a header from Conor Gallagher’s corner.

Another former City player, Raheem Sterling, put Chelsea 2-1 up in the 37th by tapping in Reece James’ cross after a sweeping attack,

The score was even again in first-half stoppage time when Bernardo Silva crossed for Manuel Akanji to head in.

City re-took the lead two minutes after the break, with Haaland scoring again — sliding in to convert from close range.

Erling Haaland restores City's lead ???????? ???????????????????????????? after half-time! ⚡️



The Norwegian secures his 13th goal of the Premier League campaign in one of the games of the season so far! ????
</item 47>

<item 48>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
What is the style of play, what’s the plan?

“He needs to take a step back, ten Hag, and say what do I need to do with these players I have available in order to first of all stop leaking goals, because
that’s a problem, and secondly, score goals – how do I get the next goal? Four games without a goal, it’s a long time.”

Advertisement

DEADLOCK BROKEN! West Ham lead against Manchester United! ????



That assist from Lucas Paquetá though… ????



Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/Qqq7zugQmj — Optus Sport (@OptusSport) December 23, 2023

Ten Hag tried to put a positive spin on the performance by claiming to BBC that they were the better side before the first goal.
</item 48>

<item 49>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
In the final year of his contract at Parc des Princes, Real Madrid are favourites to secure a free transfer next summer, but they face competition from
Liverpool.

PSG have no games left to play in 2023 due to the French winter break, with Mbappe therefore unable to catch Ronaldo.

8 Mbappe has enjoyed a stunning 2023 having scored 52 goals for club and country Credit: AFP

2*. Harry Kane (Bayern Munich/Tottenham/England) - 52 goals

A stunning 2022/23 season saw Bayern splash out over £100m to sign the Spurs legend in August.

And Kane has repaid the Bundesliga club's faith having broken a number of records with his 25 goals so far this season.

The leading scorer [22 goals] in the German top-flight, the Three Lions skipper set a new Bundesliga record last month by finding the back of the net 17 times
in his opening 11 outings.
</item 49>

<item 50>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
With two games left for City in 2023, Haaland has time to be the overall top scorer, though it remains to be seen if Pep Guardiola will play him against Everton
and Sheffield United.

Haaland has missed City's last five games with a foot injury, but it is believed he could be in contention now to return to action.

8 Haaland is the leading goalscorer in the Premier League this season

2*. Kylian Mbappe (Paris Saint-Germain/France) - 52 goals

Paris Saint Germain's all-time record scorer with 233 goals, it's easy to forget that the ex-Monaco man is still just 24 years old.

He has added even more goals to his tally for Les Parisiens this term, while he has fired in ten times for the French national team.

Mbappe's overall tally of 46 goals for Les Bleus is just ten shy of Giroud's current record.
</item 50>
=== END ITEM id=34d6a8444f4c1c0e ===
