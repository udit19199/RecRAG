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

=== ITEM id=6804eee2d2e29f4e ===
Gold sentences and facts:
Sentence 0: In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.
  Fact 0.0: The Wallabies had a World Cup flop.
  Fact 0.1: The Queensland Rugby Union reaffirmed its position.
  Fact 0.2: The Queensland Rugby Union is committed to moving to a centralised high-performance model.
  Fact 0.3: The Queensland Rugby Union will not follow the NSW Waratahs by handing over the keys to Rugby Australia.
Sentence 1: He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.
  Fact 1.0: A person will be tasked with integrating the programs.
  Fact 1.1: RA is trying to centralise all of the Super Rugby's high-performance programs.

Retrieved items:
<item 1>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU also expressed its satisfaction that RA was willing to discuss the annual $1.7m being reinstated, which had been withheld because of the Covid pandemic
to keep the game afloat.

Meanwhile, The Roar understands the ACT Brumbies have sent RA confirmation of solvency for the next 12 months.

RA was forced to take out a $40m loan from Ares to keep the game afloat.

Advertisement

While RA has around another $15m left to spend from the loan, the governing body is looking to borrow up to another $60m so they’re in the strongest possible
position to capitalise on the two major money-making events – the 2025 British and Irish Lions series and home men’s World Cup in 2017 – and succeed.
</item 1>

<item 2>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU confirmed that on Wednesday morning.

Advertisement

“The QRU is committed to working with Rugby Australia to lead reform of our game at a high-performance level, and help Rugby Australia develop an improved game
through high-performance centralisation,” the statement read.

“Centralisation” would include a national player-contracting model, and alignment of key high-performance staff, such as coaching, strength and conditioning,
and medical to operate as one national system.”

The letter pointed to RA’s involvement in Les Kiss’ appointment as Reds coach, with the former State of Origin winger turned coach, who has spent the best part
of two decades coaching in Ireland and England, replacing Brad Thorn.

As The Roar has previously reported, the QRU stopped short of wanting to hand over its commercial property.

“While the QRU is very supportive of high-performance alignment, it will not agree to any proposal on centralising commercial or corporate functions,” the
letter read.
</item 2>

<item 3>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
“The very strength of our State unions is the expertise they bring in their own markets.

Advertisement

“We are open to work on commercial initiatives together but will always retain our independence for the benefit of Queensland Rugby, its members and
stakeholders.

“Queensland Rugby has built a sustainable business model through the elimination of legacy operating debt, delivered four-straight operating profits and worked
in partnership with Governments to successfully deliver $31 million of improvements to Ballymore through the construction of the National Rugby Training Centre,
the home of the Wallaroos and Queensland Reds.”

The letter also pointed to the 71,000 players across the state, including more than 16,000 of those female participants, 257 schools featuring rugby as part of
its program, as well as its Hospital Cup competition and regional club competitions, as evidence that the game was in good health in Queensland.
</item 3>

<item 4>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The Melbourne Rebels are expected to follow.

RA chief executive Phil Waugh says building trust back between the governing body and states is essential to their plans.

Under fire Wallabies coach Eddie Jones also advocated for Australian rugby to move to a centralised model, believing the nation had fallen behind the rest of
the world.

“We’re a state-based union that’s run by the states, the national union wants to assume more responsibility and when things were going well, no one questioned
that,” Jones said.

Advertisement

“But over the last period of time, they haven’t been going so well and this tournament’s probably been the icing on the cake where it’s got to a stage where I
think people, most people probably think there’s time for change. But it’s not going to go easy.

“[It’s] one of the key issues. We need to maximize the development of the players.
</item 4>

<item 5>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised
high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.

They also expressed their strong support for Rugby Australia to conduct a “fully independent and transparent top-to-bottom review of the Wallabies program and
how it can be improved” in light of the historic first World Cup pool exit.

They added: “It should also provide a clear view of what is needed in a leadership and governance framework to ensure we put the right people, in the right
roles, in the right system.”

Facebook Twitter Whatsapp Reddit Email Share

The QRU has also expressed their desire for Rugby Australia to reinstate the annual $1.7 million in funding to each of the Super Rugby franchises that was
withdrawn following the Covid pandemic to help keep the game afloat.
</item 5>

<item 6>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
RA is also in the early stages of looking to renew its next broadcast deal, with the governing body hoping to at least double its current deal with Nine
Entertainment/Stan Sport, which is seeing them paid $29m per year until the end of 2025.
</item 6>

<item 7>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

“What’s changed back there that I could think that, what I can offer, can change what they’re doing still?

“I’ll always love rugby. I’ll always love Australian rugby. Could I see myself working in Australia again? I doubt it.”

But RA have been making moves to try and twist Nucifora’s arm.

While Nucifora was never in the running for the director of high-performance, it’s understood they are trying to get him on board as a consultant when he wraps
up with Ireland.
</item 7>

<item 8>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.
</item 8>

<item 9>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Horne briefly worked at Saracens as a general manger.

RA always said they wanted to have the role filled by Christmas, with chief-executive Phil Waugh saying the successful candidate had one of the biggest jobs to
change the game’s standing in Australia.

“It’s a big job,” Waugh said. “It’s the biggest job on my leadership team in terms of the impact it has on the general sentiment around the game.”

Horne will oversee all of Australia’s men’s and women’s high-performance programs.

Advertisement

He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.

Perhaps most pressing will be building a list of coaching candidates to take over from Eddie Jones, with RA wanting to have the new Wallabies coach set in stone
by the start of Super Rugby.
</item 9>

<item 10>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the open letter to their members on Wednesday morning, QRU chief executive David Hanham and chairman Brett Clark moved to update constituents on the reform
measures that RA is attempting to usher through the game.

Last week, The Roar revealed that the New South Wales Rugby Union was in the latter stages of handing over their license to RA.

By doing so, the NSW Waratahs, who were on the brink of insolvency, handed over their commercial and high-performance property.

Advertisement

The decision will see all male and female Waratahs players, including coaches and high-performance staff, as well as all commercial and operations staff, run
out of RA’s office.

NSW Waratahs chief executive Paul Doorn informed stakeholders of the decision last week.

RA is cognisant that not every Super Rugby franchise will follow suit immediately, but the governing body is hopeful that now one of the two biggest states has
agreed to hand over the keys, others will follow suit swiftly.
</item 10>
=== END ITEM id=6804eee2d2e29f4e ===

=== ITEM id=f9f40af23c36150a ===
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
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
Not enough conversation.”

There were other things that may have hurt Pebble’s traction, too, including its lack of a native mobile app. As T2, the startup prioritized building for the
web to be agile and more flexible, but lost out on the opportunities for discovery driven by the app stores. Notes Greer, “that might have been a mistake.”

In retrospect, Cselle also questions if the rebranding from T2 to Pebble may have been an issue.

“I’m wondering if the Pebble rebrand had actually softened the value prop[position] because T2 just encapsulated it so much — that it was really duplicative [of
Twitter],” he says.

Like most startup shutdowns, the answer as to “why” is not any one thing but a combination of factors. For Pebble, that’s certainly true.
</item 1>

<item 2>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
For Pebble, that’s certainly true. It was a perfect storm of competition, X’s continued traction, the lack of a native app, a brand that didn’t resonate and a
space that was maybe a little too safe to be as addictive and as fun as the original.

With Pebble’s winddown, being announced to users today at 9 AM PT, early adopters will have the option of exporting their Pebble archive as a zip file that
loads a mini web page showcasing all their old posts. Pebble won’t be directing users back to X however, or any other social network.

As for where the founders will now hang out?

“I don’t know. Maybe LinkedIn,” Cselle jokes.

Pebble is also returning a small amount of the funds left over to its investors, more as a courtesy to show financial responsibility. The Pebble website will be
shut down on November 1st and there aren’t further plans for the IP at this time.
</item 2>

<item 3>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think Threads and others have encountered this as well,” adds Pebble co-founder and CTO Michael Greer, previously Discord engineering head. “The durability
of the network effect of Twitter is stronger than anticipated,” he says.

Despite these challenges, Pebble initially had decent retention. By week four, it was seeing retention rates of 30%, in fact. Its invite list was also working,
as 60% of people who received a Pebble invite converted to become users. In addition, some 10,000 users arrived at that list from early press, like TechCrunch’s
coverage of its first outside funding — a $1.1 million angel round that included investors like former Google VP Bradley Horowitz, Android co-founder Rich Miner
and the former CEO of Wikipedia, Katherine Maher. Data from Similarweb indicates the Pebble website topped 99,300 visits last month, indicating repeat usage.

The company’s hypothesis was that consumers were hungry for a Twitter alternative that prioritized trust, safety and moderation from the start.
</item 3>

<item 4>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The founders don’t regret their experience building Pebble, even though it didn’t turn out as they hoped, they said.

Says Cselle, “One thing that I learned is that there’s absolutely an audience that wants to see a new kind of Twitter-like platform built and will ask for the
features that Twitter has.” He suspects the team might stay together to work on something new after Pebble wraps, taking the learnings from Pebble with them.
That includes better communicating to users what great content should look like on a platform, and how they can win.

“We’re at an inflection point in social media,” says Oh. “We had one hypothesis with T2 and Pebble. It obviously didn’t pan out the way that we wanted it to.
But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
</item 4>

<item 5>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
To that end, Pebble’s founding team included co-founder Sarah Oh, Twitter’s former human rights advisor.

Pebble continues to believe its approach to moderation was correct, even if it was not ultimately a factor in driving growth.

“We came in with a particular angle: kinder, safer. Trust and safety. And I think our approach to moderation that we did have on the site did work,” says
Cselle, adding that Pebble didn’t have some of the issues other federated platforms faced. For example, TechCrunch reported that Bluesky earlier this year had
strained its relationship with Black users, as it failed to crack down on bigotry on the site. Later, users on Bluesky were even creating usernames with racial
slurs, and yet the platform didn’t issue a public apology. Pebble never faced any such controversies.

But it may have positioned itself a little too far into the “kindness” territory, the founders now believe.
</item 5>

<item 6>
Pittsburgh boy, 10, needs second liver transplant to save his life: ‘Only possible through love’ (FOX News - Health, 2023-12-09T16:31:40+00:00)
Dr. George Mazariegos, chief of pediatric transplantation at UPMC Children's Hospital of Pittsburgh, said Lucas is experiencing some of the more subtle changes
of liver failure.

PIG HEART RECIPIENT'S ‘LAST WISH’ WAS TO HELP DOCTORS LEARN FROM PROCEDURE, DYING MAN SAID: ‘WE WILL MISS HIM'

"Those include not being able to grow normally and not getting the proper levels of oxygen to his body and brain, because the blood is being shunted away from
the liver," he told Fox News Digital.

"Those are important and potentially life-threatening complications."

As of now, Lucas is still at home and "managing his symptoms," Mazeriegos said, but the doctor emphasized the importance of his receiving a timely transplant
before further complications occur.

next Image 1 of 2

prev Image 2 of 2

The Goellers are now seeking support again, launching the "Save Lucas" campaign across social media.
</item 6>

<item 7>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think the competitive landscape evolved faster than we had thought,” he explains. “I didn’t think that quite as many people — established organizations and
newcomers — would try to do the same thing that we were doing and in very similar ways.”

Today, the market for Twitter alternatives is a crowded one, to say the least. There are numerous services for consumers to try, including the open source-based
platform Mastodon, a soon-to-be decentralized system from Bluesky, plus smaller startups like Spill, Spoutible and Post, as well as a new app from Meta called
Instagram Threads.

Then, of course, there’s the X factor to consider. The default platform for short posts retains some pull, despite — or perhaps because of — new owner Elon
Musk’s antics. Though the company may not have yet succeeded from a business and financial perspective, it’s been hard for others to duplicate its function as a
breaking news platform and place for spirited debates.
</item 7>

<item 8>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The grip Twitter, now called X, has on the market may be stronger than some believed. Unfortunately, that’s led to the first casualty among Twitter
alternatives, as the startup Pebble (formerly T2), is shutting down. The would-be X rival had grown a small but engaged community on its microblogging service
that aimed to dupe Twitter’s features, from its verification systems to functionality like DMs. But the company says they’ve run out of time to make Pebble
happen — the app maxed out at 3,000 daily active users, out of 20,000 registered users. That daily user figure fell to 1,000 daily users following its
rebranding from T2.

In part, the sizable competition coming from other Twitter rivals was to blame, says co-founder and CEO Gabor Cselle, who had exits to Google and Twitter prior
to starting Pebble.

“I think the competitive landscape evolved faster than we had thought,” he explains.
</item 8>

<item 9>
Pittsburgh boy, 10, needs second liver transplant to save his life: ‘Only possible through love’ (FOX News - Health, 2023-12-09T16:31:40+00:00)
And the city of Pittsburgh has rallied once more behind the family, with newly donated billboards and media campaigns.

"We're trying to not only save Lucas, but also to build organ donation awareness for both deceased and living donors," said Goeller.

ULTIMATE FATHER’S DAY GIFT: SON DONATES KIDNEY TO SAVE HIS DAD’S LIFE

"We want it to really sink into hearts around the world because it's so important."

Despite his challenges, young Lucas Goeller remains positive, active and upbeat, continuing to enjoy his love of the outdoors and hunting with his father and
his three brothers, aged 12, 8, and 4.

"He's always filled with so much joy," his mother said. "We get our strength from him. He encourages all of us to continue to look at the positives in life."

The ideal living donor for Lucas is someone with type O blood, aged 20 to 49, with a BMI below 30.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=f9f40af23c36150a ===

=== ITEM id=0e1d49a2e6ba009d ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their liability.
  Fact 1.0: Sportsbooks might adjust the odds to balance their liability if a significant amount of money is placed on one team or participant.

Retrieved items:
<item 1>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
However, you’d have a winner at the fourth. If you didn’t shop around, you’d never have found that half-point victory and be stuck with a push instead of a
profit.

What is the Vegas line?

Las Vegas sportsbooks set the standard for lines and odds across the country. If a sportsbook is copying a line, it’s usually copying the Vegas line. Most
online betting sites don’t stray too far from the Vegas line, using it as a foundation for setting all lines in every major sport.

If those lines do shift, it’s usually because of increased betting on one side of a bet. Shop lines locally, and you can take advantage when sportsbooks stray
from the Vegas line.

Why do the lines move?

It’s worth reiterating that sportsbooks want an equal amount of money bet on each side of every bet. When that happens, they earn the small profit built into
the odds without concern about who wins or loses.
</item 1>

<item 2>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 2>

<item 3>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
This includes your initial $10 wager, along with a profit of $15.30.

It's essential to note that this calculation gives you the total payout, not just your earnings. If you're keen on tracking your profits from betting, remember
to subtract your original stake from the payout to get an accurate figure.

Why do moneyline odds change?

Lines at the best sports betting apps change quickly due to two primary reasons:

Public betting

New information

Before setting odds for an event, oddsmakers gather extensive data. Once they finalize the odds, they release them to the public.

As a bettor, you can bet immediately or wait for potential shifts in the odds, although predicting these shifts is challenging.

For instance, when examining odds for the next Super Bowl champion released shortly after the previous Super Bowl, these odds are based mostly on the recently
concluded season.

As the new season approaches and teams make changes, odds adjust based on perceived championship chances. As games progress, lines continue to evolve based on
team performances.

Public betting can also sway lines.
</item 3>

<item 4>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 4>

<item 5>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks want even betting so much that they’ll move a line or change the odds to draw more bets to the other side to get it that way.

That’s when you’ll see a difference between the lines and odds. Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like
an injury or roster move. It’ll be difficult for you to find value in that case.

But when a line moves or odds change because of betting patterns, only the sportsbook that saw the betting makes the change. That’s your opportunity right
there.

Line shopping is a tool to help you spot changes due to betting patterns. Then, you can simply take advantage of movement in a line or changes in the odds.

Choose a side, shop around at online sportsbooks, compare the lines and odds, and then pounce when you find one that has something different and more favorable
to your viewpoint.

See more: What is a parlay bet?
</item 5>

<item 6>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.

Safety and Security:

Top NBA betting sites prioritize users' safety.

Ensure your personal information is secure.

Some sportsbooks excel in ensuring cybersecurity; always go for those with solid reputations.

How to find the best NBA lines

Not all sportsbook betting odds are created equal. If you're willing to put in some legwork, you can shop for the best NBA lines to maximize your potential
return.

Movement in the odds depends on multiple factors, including team news, injuries, public betting, recent play, and even the number of games a team plays in a
row.

Oddsmakers use all the available information to craft their lines, and you can find value in shopping around for the most favorable lines and odds.

You can use our NBA odds page (see link above) to find the best lines for your prospective bet, or you can open accounts at multiple sportsbooks and do the
shopping yourself.
</item 6>

<item 7>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Does overtime count in my moneyline bet?

Yes, in most sports and with most sportsbooks (including new betting sites), overtime (or any extra time or tiebreakers) does count in a moneyline bet.

That said, always check the specific rules of the sportsbook you're using to ensure you understand their policies.

Why do moneyline odds change?

Moneyline odds change for several reasons.

Public Betting Patterns: If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their
liability. By making the odds less favorable for a heavily-bet team and more favorable for the less-bet team, they can encourage more even betting.

If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their liability. By making the odds
less favorable for a heavily-bet team and more favorable for the less-bet team, they can encourage more even betting.
</item 7>

<item 8>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on one side might indicate that the
initial odds were potentially "off," prompting an adjustment. Event Dynamics: For live betting, where bets are placed while the event is in progress, the
evolving dynamics of the match can lead to frequent changes in the odds.

Sportsbooks adjust moneyline odds to ensure they remain profitable, balance their book, and reflect the most current information about the event.

Related Pages:
</item 8>

<item 9>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
However, the only way to find these opportunities is shopping lines.

See more: NFL futures bets

Can I get a better line or odds after I’ve already bet?

When you bet, you lock in the odds or line as currently posted. If they change, for better or worse, you’re stuck with the odds from when you wagered.

That’s precisely why you should go line shopping before you bet. Of course, if significant changes occur, you can take more action at the improved odds or more
favorable line.

Is line shopping worth my time?

Because it can provide you the opportunity to lay less, win more, and improve your chances, line shopping is certainly worth your time. Plus, it doesn’t have to
take up all that much of it.

Use the live odds feed above to do a basic survey. You’ll find it takes just seconds to compare lines and odds. Plus, download and install multiple licensed
online sportsbook apps on your phone, and you can do a more thorough shop in just a few minutes.
</item 9>

<item 10>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Do point spread odds change?

Yes, point spread odds can change, and these shifts are commonly referred to as "line movement."

Lines move when the point spread or the odds associated with a particular bet change after sportsbooks have initially set them. There are several reasons for
these adjustments.

One of the most common causes is an imbalance in the amount of money being wagered on one side of the bet. If a significant majority of bettors are placing
their money on one team to cover the spread, the top sports betting apps will adjust the point spread to make the other side more appealing, thereby balancing
the betting action.

Another factor that can affect the spread odds is the occurrence of unexpected events, such as injuries to key players, changes in team strategies, or even
significant weather events that might impact a game's outcome.

For instance, if a star quarterback is announced as injured just days before a big game, the point spread might shift in favor of the opposing team.

Moreover, information and public perception play a role.
</item 10>
=== END ITEM id=0e1d49a2e6ba009d ===

=== ITEM id=f747f9bb2bd50a83 ===
Gold sentences and facts:
Sentence 0: Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto.
  Fact 0.0: Shares of Rogers extended losses on news of the lawsuit.
  Fact 0.1: Rogers shares closed down 2.2% to C$53.07 in Toronto.
Sentence 1: Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it.
  Fact 1.0: Amazon tumbled 4 per cent after the Federal Trade Commission filed an antitrust lawsuit against it.
  Fact 1.1: 17 state attorneys general filed an antitrust lawsuit against Amazon.
  Fact 1.2: The Federal Trade Commission and the 17 state attorneys general filed the antitrust lawsuit against Amazon together.

Retrieved items:
<item 1>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Hart said Amazon sales comprise about 90% of his business, meaning any hit those sales take on Amazon has a considerable impact. He recalled “24 anxious hours”
in September 2022 when a third-party seller of his popular construction toy Brain Flakes listed the toy for a lower price on Target than it was offered on
Amazon.

Molson Hart, CEO of Viahart, an educational toy company that sells on Amazon. Courtesy: Molson Hart
</item 1>

<item 2>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The documents show that the fractures inside the prominent Toronto family have never healed. Rogers Communications is the country’s largest wireless provider
and the owner of a cable television empire, the Toronto Blue Jays baseball team and other media and sports assets. The family’s controlling stake in the public
company is worth close to C$8 billion ($5.9 billion), according to calculations based on company filings for its 2023 annual meeting.

Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto. They’ve dropped 16% this year.

The children of late founder Ted Rogers have frequently fought one another — sometimes driving down Rogers’ stock price as investors grew worried about the
company’s governance.

The bad blood spilled into full view in October 2021, when Edward Rogers publicly battled his sisters and his mother for control of the board and won in court.
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers.
</item 2>

<item 3>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The Shaw takeover finally closed on April 3 of this year, and the Rogers siblings engaged in talks “intended to resolve all issues between us,” Rogers-Hixon
stated in an affidavit. “I was hopeful all issues could be resolved without negative publicity.” But those negotiations collapsed last month, she said.

It isn’t clear from the court filings whether the discussions included the possibility that any Rogers family members would sell their stake in Rogers
Communications, leave their roles at the company, or exit the family trust that controls almost all of the voting shares of the public company.

“This matter should be resolved privately,” Rogers Communications spokesperson Sarah Schmidt said by email. “We’ve demonstrated that we will not be distracted
by these actions — we have significant momentum in the market, our merger is tracking ahead of plan, and we remain squarely focused on doing what’s right for
our customers and stakeholders.”
</item 3>

<item 4>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.
</item 4>

<item 5>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Two members of the billionaire family that controls Rogers Communications Inc. are suing the company, saying they’ve been frozen out of board meetings and
blocked from getting information, reigniting a bitter power struggle within one of Canada’s wealthiest clans.

Melinda Rogers-Hixon says she’s the target of a “personal vendetta” by her older brother, Rogers Chair Edward Rogers, according to a letter contained in the
court filing. The Rogers siblings were close to an agreement that would have resolved their longstanding differences, but Edward Rogers reneged on it, according
to the Sept. 20 missive by Rogers-Hixon, which was sent to Chief Executive Officer Tony Staffieri.

The court petition was filed by Rogers-Hixon and her sister, Martha Rogers, and it argues the company has been “oppressive” to them.

The documents show that the fractures inside the prominent Toronto family have never healed.
</item 5>

<item 6>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers. Soon after, then-CEO Joe Natale was sacked and
replaced by Staffieri, the former chief financial officer. Rogers-Hixon and Martha Rogers had opposed the CEO change.

In early 2022, family members agreed to temporarily shelve their disagreements, according to this week’s court filing, to avoid negative publicity as the
company sought regulators’ approval for the takeover of Shaw Communications Inc. The C$20 billion transaction was the company’s biggest deal ever and among the
largest corporate transactions in Canadian history.

Negotiations Collapse

As part of that truce, Rogers-Hixon and Martha Rogers agreed not to participate in board discussions about the Shaw deal or receive materials on it, because of
an argument about Rogers-Hixon’s personal lawyer, whose firm also does work for Telus Corp., a competitor to Rogers and Shaw.
</item 6>

<item 7>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.
</item 7>

<item 8>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.
</item 8>

<item 9>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Just like any store owner who wouldn’t want to promote a bad deal to their customers, we don’t highlight or promote offers that are not competitively priced.”
Zapolsky argued the FTC’s lawsuit could force it to stop highlighting low prices, “a perverse result that would be directly opposed to the goals of antitrust
law.”

“Long overdue” lawsuit

On Amazon’s own forum for merchants, called Seller Central, several users cheered on the FTC and said they hoped it would result in changes to the company’s
business practices. Amazon’s tense relationship with merchants has been well-chronicled over the years, with sellers expressing a range of grievances over
issues like rising fees, an arcane suspensions process, and heightened competition on the marketplace from all sides, including the e-commerce giant. “I think
it’s great, Amazon deserves it,” one person commented, adding, “More should be coming on the way.”
</item 9>

<item 10>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Big Tech stocks tend to be among the hardest hit by high rates, and they were the heaviest weights on the index. Apple fell 2.3 per cent and Microsoft lost 1.7
per cent. Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it. They accuse the
e-commerce behemoth of using its dominant position to inflate prices on other platforms, overcharge sellers and stifle competition. In China, concerns continued
over heavily indebted real estate developer Evergrande. The property market crisis there is dragging on China’s economic growth and raising worries about
financial instability. France’s CAC 40 fell 0.7 per cent, and Germany’s DAX lost 1 per cent.

Crude oil prices rose, adding to worries about inflation. A barrel of benchmark US crude climbed 71¢ to $US90.39. Brent crude, the international standard, added
67¢ to $US93.96 per barrel.
</item 10>
=== END ITEM id=f747f9bb2bd50a83 ===

=== ITEM id=46d8352b3688fdb5 ===
Gold sentences and facts:
Sentence 0: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 0.0: Google unveiled new hardware.
  Fact 0.1: Google held its annual hardware event this week.
  Fact 0.2: Google is a search and consumer tech giant.
  Fact 0.3: At its annual hardware event, Google showed off what it has been working on.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed the article for not meeting its standards.
  Fact 1.1: The article was shared widely on social media.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The extent to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google "siphons off" news publishers' readers.
  Fact 2.3: The case argues that Google "siphons off" news publishers' ad revenue.
  Fact 2.4: The case argues that Google obtains these through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 4>

<item 5>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 10>
=== END ITEM id=46d8352b3688fdb5 ===
