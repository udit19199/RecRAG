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

This file has 6 items.

=== ITEM id=961fb5ebf6f2cdcf ===
Gold sentences and facts:
Sentence 0: Even so, Kelce's 81.3 average receiving yards per game against the Raiders since 2019 suggests the All-Pro tight end is in line for a productive outing and could potentially snap his four-game scoring drought.
  Fact 0.0: Kelce averages 81.3 receiving yards per game against the Raiders since 2019.
  Fact 0.1: Kelce is an All-Pro tight end.
  Fact 0.2: Kelce's average receiving yards against the Raiders suggest he is in line for a productive outing.
  Fact 0.3: Kelce could potentially snap his four-game scoring drought.
Sentence 1: The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the classic game “Kiss, Marry, Kill”.
  Fact 1.0: The first link between Swift and Kelce was established in 2016.
  Fact 1.1: In 2016, Kelce played the classic game "Kiss, Marry, Kill" and that is when the first link between Swift and Kelce was established.
  Fact 1.2: Kelce helped bring his team to Super Bowl victory in 2020.
  Fact 1.3: Kelce helped bring his team to Super Bowl victory in 2023.

Retrieved items:
<item 1>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can Raiders QB Aidan O’Connell earn next season’s starting job?: After leading the offense on seven touchdown scoring drives in last week's 63-21 win over the
Chargers, the rookie fourth-rounder aims to further establish himself as a serious contender for Las Vegas' starting job next season. Despite the encouraging
performance, O'Connell has been inconsistent across six games as the full-time starter, throwing for 1,300 yards, seven touchdowns, and five interceptions.

However, a stellar showing against the NFL's fifth-ranked pass defense (202.5 yards per game allowed) would benefit his bid to be the starter in 2024.

Chiefs TE Travis Kelce: Yes, his girlfriend will be in town for the holiday. Kelce is averaging 89.6 yards per game with global popstar Taylor Swift in
attendance this season but is coming off his worst outing in her presence, catching five passes for 28 yards.
</item 1>

<item 2>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Well I would say that Travis Kelce's had a lot of big catches in his career. This would be the biggest."

Sept. 26: Chiefs games get even more popular

"Welcome to New York (*New Jersey), it's been waiting for you." - Welcome to New York

The defending Super Bowl champions are already one of the more popular teams in the league, but it is becoming the hottest ticket in town, no matter what town
that is.

"Since Swift's appearance at the Chiefs game on Sunday, the Jets have sold more tickets for their game against the Chiefs in a single day than since the season
started -- by more than double," a representatives at StubHub told to Town & Country magazine.

Sept. 27: Mahomes confirms he met Swift

"It was enchanting to meet you, all I can say is I was enchanted to meet you."
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.
</item 4>

<item 5>
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
</item 5>

<item 6>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.
</item 6>

<item 7>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the
classic game “Kiss, Marry, Kill”. Speaking with AfterBuzz TV, Kelce was given the options of Swift, Katy Perry and Ariana Grande. He said he’d ultimately kill
Grande, marry Perry and kiss Swift. That was about six years ago, but it has more recently set tongues wagging as fans wondered how long the plan for them to
meet has been in the works. Cruel Summer Fast-forward seven years, and things started to heat up. In July, speaking on the podcast he shares with his brother,
New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City. Unfortunately, the
bracelet never made it to her wrist or his digits to her phone.
</item 7>

<item 8>
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
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 10>
=== END ITEM id=961fb5ebf6f2cdcf ===

=== ITEM id=b1308bc72ce13eae ===
Gold sentences and facts:
Sentence 0: Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.
  Fact 0.0: Depending on which one you get, it might come with USB ports.
  Fact 0.1: The USB ports let you charge multiple devices in one outlet.
Sentence 1: With a 10,000-mAh capacity, WIRED contributor Simon Hill says it charged his iPhone 14 Pro once and then delivered an additional 64 percent the second time before tapping out.
  Fact 1.0: The device has a 10,000-mAh capacity.
  Fact 1.1: WIRED contributor Simon Hill says the device charged his iPhone 14 Pro once.
  Fact 1.2: Simon Hill says the device then delivered an additional 64 percent to his iPhone 14 Pro the second time.
  Fact 1.3: Simon Hill says the device tapped out before the second charge was finished.

Retrieved items:
<item 1>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
It fluctuates in price very often, but it's still a good deal.

Satechi's fold-up wireless charging stand has a 10,000-mAh capacity (complete with LEDs to show you how much juice you have left) and can wirelessly charge both
your phone and a wireless earbuds case (as long as it's Qi-certified). There's also a USB-C port to power up a third device. The only gripe we have is that it's
slow—delivering 10 watts of wireless charging power for phones (7.5 watts for iPhones), 5 watts for earbuds, and 10 watts via the USB-C port.

Apple 35-Watt Dual USB-C Port Compact Power Adapter Photograph: Apple

Apple's dual USB-C port power adapter is an essential accessory if you have a tendency to forget to pack chargers for your devices on the go. With two USB-C
ports, you can use it to recharge both your iPhone and MacBook simultaneously.
</item 1>

<item 2>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Older buildings especially suffer from what I can only describe as loose socket syndrome, where those very convenient modern boxy socket extenders with five
different USB and USB-C charging ports simply cannot stay in place and fall right out of the wall because they’re too heavy. After one (totally pleasant)
vacation where I had to charge my phone, watch, AirPods, and laptop in a weird corner of my room with the plug-in charging hub propped up on a strategically
balanced mountain of books and sham pillows, I bought this Anker combination extension cord / power strip. It’s not a full power strip — just a cube with a few
sockets along with USB and USB-C charging ports at the end of a five-foot cable. I’ve brought it on a few trips since then. It takes up extra space in my
suitcase but each time has left me feeling vindicated about the purchase.

The three prongs at the end of the cable are static, rather than folding flat for easy packing.
</item 2>

<item 3>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It has a little stand to prop it up, a short cord that goes out to an outlet, and four USB-C ports that output 100 watts each. This is the lowest price we've
tracked.

Click the coupon button to see the discount at checkout. We recommend this Baseus charger if you're looking for a high-capacity option. With a 10,000-mAh
capacity, WIRED contributor Simon Hill says it charged his iPhone 14 Pro once and then delivered an additional 64 percent the second time before tapping out. It
comes with a USB-C cable too. The 6,000-mAh version is also on sale for $30 ($20 off). These deals are only applicable to Amazon Prime members.

Apple 35-Watt Dual USB-C Port Power Adapter Photograph: Apple

If you tend to forget to pack chargers for your devices, Apple's dual USB-C port power adapter is an essential accessory.
</item 3>

<item 4>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Granted, it’s bulkier than buying one or two specialized adapter plugs, but if you’ve got a multi-continent itinerary, it’s a game-changer. What I like about
this one from Epicka is that it comes with a spare fuse in case things go sideways with voltage.

The only caveat is that, although it says “universal,” it’s technically only the four most common types of plugs. That’ll get you by in most countries, but it’s
not a guarantee in places like Brazil, South Africa, or India. Even so, I’ll take this over price gouging at airport electronics shops or having to take time
out of my schedule to visit a local hardware store.

Lovely lights

Brandon Widder, senior commerce editor

I’m a big fan of ambient lighting, even when I’m 50 miles from the nearest outlet.
</item 4>

<item 5>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
But that’s just life while traveling with a dog anyway.

An extension cord for awkward situations

Sarah Jeong, deputy features editor

Anker 321 Power Strip $ 15 $ 26 42 % off $ 15 $ 15 $ 26 42 % off This all-in-one 20W USB-C power cube boasts three AC outlets, two USB-A ports, and one USB-C
port. $15 at Amazon

Nobody wants to carry a power strip or an extension cord with them on their vacation. It’s probably unnecessary if you’re staying in relatively modern buildings
and definitely unnecessary if you’re camping. But sometimes you want to stay in a charming historical hotel or a lovely cabin in the woods, and it’s only when
you go to charge your devices at night that you realize that the only electrical socket in the bedroom is in the corner farthest away from the bed and there’s
already two lamps plugged into it.
</item 5>

<item 6>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.
</item 6>

<item 7>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
There's even a built-in display to monitor stats like temperature and remaining battery percentage.

This heavy metal brick is a good option for charging laptops and smaller gadgets. It packs 20,000 mAh and supports a wide variety of fast charging standards.
There is one USB-C PD port rated at 65 watts, two USB-A QC ports at 30 watts apiece, and a micro-USB input (though you are best using the USB-C to recharge it).
We haven't seen it go on sale very often.

This travel kit from ESR works well if you want something that can prop your iPhone in portrait or landscape orientation. It can also display your Apple Watch
in Nightstand mode, so you can peek and see how long before you must get out of bed. The main body folds open with a MagSafe charging pad for your iPhone and a
slot behind for your AirPods. We have seen this dip a little lower, but this is still a solid deal.
</item 7>

<item 8>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Flip over to the back, and you'll find two USB-A ports, two USB-C ports, and three AC outlets. It's a compact and tidy charger, and our unit has been going
strong for more than two years. It fluctuates in price very often, and it's hit as low as $60 in the past, but we still think this is a good deal.

This power bank is one of our favorites if you're looking for an upgrade. It has an absolutely gargantuan 24,000-mAh capacity, and it can be charged up in an
hour. It can output up to 140W, making it powerful enough to charge laptops and tablets, as well as charging your phone super quick.

This is a great charger if you travel often with an iPhone and Apple Watch. Instead of packing two separate chargers, the Duo offers the luxury of both all
within a compact, foldable form factor.
</item 8>

<item 9>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Universal travel adapter

Victoria Song, senior reviewer

Epicka universal travel adapter $ 20 $ 25 20 % off $ 20 $ 20 $ 25 20 % off Epicka’s universal travel adapter is an all-in-one adapter that includes four
different plugs that cover over 150 countries. $20 at Amazon

In my youth, I forgot to pack plug adapters for international trips one too many times. Buying them once you’ve landed in another country isn’t always easy,
either. And if you’re like me, your relatives in rural Korea don’t always have more than one plug for your American devices — in which case, you’ll have to
share with your six other cousins. Nope. Absolutely not. Which is why I never leave this country without a universal travel adapter.

Basically, it’s six plug adaptors in one. Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.
</item 9>

<item 10>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
The performance is fast, the recharging is fast. It even looks fast. It has a bright, 120-Hz AMOLED screen, great speakers, and surprisingly good battery life
for all that. Our main gripe is the IP64 water- and dust-resistance rating, which is not nearly as good as other flagship phones.

Ugreen's charger is, as the name suggests, a 145-watt charger with a 25,000-mAh battery. It's surprisingly compact for the power it provides, although at 1.1
pounds, it's definitely not ultralight. There are two USB-C ports and one USB-A port. What sets the Ugreen apart is that you can actually draw 145 watts while
charging. That works out to one USB-C port at 100W and the other at 45W. Very few other batteries we've tested are capable of that feat.

Photograph: Anker

The Anker Nano is one of our favorite portable power banks, particularly for phones.
</item 10>
=== END ITEM id=b1308bc72ce13eae ===

=== ITEM id=dbac6fffd089c06e ===
Gold sentences and facts:
Sentence 0: Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana.
  Fact 0.0: The U.S. will travel to GEODIS Park in Nashville, Tennessee on October 17.
  Fact 0.1: The U.S. will play a friendly against Ghana at GEODIS Park.
Sentence 1: Fellow import Milton Doyle and Majok Deng each scored 17 for Tasmania, who recorded their second win over Sydney in as many attempts this season.
  Fact 1.0: Milton Doyle is a fellow import.
  Fact 1.1: Majok Deng is a fellow import.
  Fact 1.2: Milton Doyle scored 17 for Tasmania.
  Fact 1.3: Majok Deng scored 17 for Tasmania.
  Fact 1.4: Tasmania recorded their second win over Sydney this season.
  Fact 1.5: Tasmania's second win over Sydney this season came in as many attempts as the team made this season.

Retrieved items:
<item 1>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 1>

<item 2>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 2>

<item 3>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 3>

<item 4>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
While Christian Pulisic starred on the left, Tim Weah was equally devastating down the right, dazzling with electric pace and good link-up.

Weah's biggest moment came when he scorched Robin Gosens on the wing in the first half, knocking it past the German defender and then beating him with
blistering speed to get back on the ball. Weah's statistical haul don't quite do him justice, but he was a threat throughout his 66 minutes of action.

ST: Folarin Balogun — 7

U.S. fans are salivating for Folarin Balogun to get more touches up front, and as Gregg Berhalter foreshadowed pre-match, they made a clear attempt to do that.
Gio Reyna and Tim Weah did the most of the work in that regard,

LW: Christian Pulisic — 9

There's no doubting who remains the most electric and incisive U.S.
</item 4>

<item 5>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 5>

<item 6>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 6>

<item 7>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
player is, as Christian Pulisic once again backed up his nickname of 'Captain America'. The first half stunner was emblematic of his performance, but beyond
that he constantly proved a threat to German's out-of-position right-back Jonathan Tah.

Pulisic completed 27 of 29 passes, a tidy haul for a player of his attacking intent, and created two chances in addition to getting the goal. His corners remain
awful, and it's a wonder why he is still on duty for that role, but otherwise Pulisic was a menace.

Christian Pulisic with an absolute banger 🚀



Watch USA vs.
</item 7>

<item 8>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Sporting News reviews how each USMNT player performed across the 90 minutes.

MORE: Recap the 3-1 defeat for the USMNT against Germany in Hartford

USA player ratings vs Germany

USA starting lineup (4-3-3, right to left): 1. Turner (GK) — 19. Scally, 4. Richards (Carter-Vickers, 66'), 13. Ream, 2. Dest — 7. Reyna (de la Torre, 46'), 6.
Musah, 8. McKennie (Cardoso, 75') — 21. Weah (Aaronson, 66'), 20. Balogun (Pepi, 66'), 10. Pulisic (Paredes, 75').
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: Breaking down Germany squad, tactics as Julian Nagelsmann debuts as manager

USA roster for October friendlies vs Germany, Ghana

The USMNT squad for the October friendlies was announced on Thursday, October 5. Gregg Berhalter has called in 23 players, which is one fewer than were brought
in for the friendlies against Uzbekistan and Oman last month.

Below is the official 23-man roster for the upcoming October friendlies against Germany and Ghana.

PSV midfielder Malik Tillman was forced to withdraw from the squad due to injury, with Alex Zendejas of Club America named to the roster in his place.

MORE: Tracking every goal scored by Christian Pulisic at new club AC Milan

Pos Name Club Caps Goals GK Ethan Horvath Nott. Forest (ENG) 9 0 GK Matt Turner Nott. Forest (ENG) 33 0 DEF Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino
Dest PSV Eindhoven
</item 10>
=== END ITEM id=dbac6fffd089c06e ===

=== ITEM id=312d6e4006dd5502 ===
Gold sentences and facts:
Sentence 0: Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a return to fitness, and has been included on the squad.
  Fact 0.0: Johnny Cardoso had to pull out of the September camp due to an ankle injury.
  Fact 0.1: Johnny Cardoso has returned to action with Brazilian club Internacional.
  Fact 0.2: Johnny Cardoso's return to action indicates a return to fitness.
  Fact 0.3: Johnny Cardoso has been included on the squad.
Sentence 1: He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.
  Fact 1.0: Johnny Cardoso and his fresh legs are back just in time.
  Fact 1.1: Rookie sensation De'Von Achane hit Injured Reserve this week.
  Fact 1.2: De'Von Achane hit Injured Reserve with a knee injury.

Retrieved items:
<item 1>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
However, even a moderate MCL sprain could mean a multi-week absence for the injury-prone veteran back, especially in a lost season for the 4-6 Pack. Rookie
Emanuel Wilson also hurt his shoulder in Week 11, so AJ Dillon could be getting the keys to Green Bay's backfield castle (although it's more of an oversized
shack, at this point).

Dolphins rookie De'Von Achane, who looked like the runaway Offensive Rookie of the Year earlier this season, looked to have aggravated his MCL injury upon his
Week 11 return against the Raiders. Achane has a chance to return for Miami's Week 12 tilt with the Jets on Friday, but we wouldn't be surprised if the 7-3
'Fins play it safe and keep him out until he's back to 100 percent. Meanwhile, Miami placed backup Salvon Ahmed on IR with a foot injury, so Jeff Wilson Jr.
could be back in play as a handcuff stash.
</item 1>

<item 2>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 2>

<item 3>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He featured his usual pressing, and came close to scoring once on a messy chance that Germany just barely managed to clear, but otherwise he didn't quite cause
the opposing back line problems to open spaces up for teammates.

75th min: Johnny Cardoso — 6

Johnny Cardoso only had 15 minutes at the end to showcase his abilities, but it begged the question why he or Lennard Maloney weren't the ones introduced at
halftime instead of Luca de la Torre. Cardoso lost possession just once in his 10 touches and completed a tackle, although he was also dribbled past a few times
as well.

75th min: Kevin Paredes — 5

With 10 touches in his 15 minutes of action towards the end of a mucked up game of numerous substitutions, Kevin Paredes didn't stand out, losing possession
four times and completing four passes. Tough to evaluate his time on the field, but he didn't stand out.
</item 3>

<item 4>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 4>

<item 5>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
For Taylor, he underwent surgery on his thumb this week and is expected to miss at least a couple of weeks, which will thrust Zack Moss into a more prominent
role in the Colts backfield.

Burks is the only Titan to have an injury designation entering the weekend, but the wideout was a full participant in Friday's practice. However, he will need
to clear the final stages of concussion protocol in order to play.

Dolphins (-9.5) at Commanders

Every player who the Dolphins listed as questionable for Sunday's game against Washington was able to practice on a limited basis on Friday. That includes
Cotton and Lamm, who both were DNPs earlier in the week. Rookie running back De'Von Achane was a full participant in Friday's practice and head coach Mike
McDaniel said he will play this week after missing time with a knee injury.

Washington could be without its starting center this weekend as it plays host to Miami.
</item 5>

<item 6>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 6>

<item 7>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.

DOLPHINS: Aren't quite the bastion of rushing that some headlines have made them out to be, but they're just over league average in rush rate at 43.7% and above
league average in RB runs per game at 22.6. All of these numbers are weighted heavily by their 50-point win over Denver in Week 3 when they ran the ball 42
times with their running backs. Their 7.6 yard per carry average is due to Achane's breakout start more than anything else.
</item 7>

<item 8>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 10>
=== END ITEM id=312d6e4006dd5502 ===

=== ITEM id=5d0fb9a9c69a58c0 ===
Gold sentences and facts:
Sentence 0: Research shows that even a small reduction in sleep has been shown to impact your immune system, however.
  Fact 0.0: Research shows that even a small reduction in sleep impacts the immune system.
Sentence 1: “This sex difference in insomnia emerges after puberty, suggesting that hormonal differences such as menstruation, pregnancy and menopause may be the cause of this increased risk in women,” Dr Browning says.
  Fact 1.0: Dr Browning says this sex difference in insomnia emerges after puberty.
  Fact 1.1: Dr Browning says the sex difference suggests hormonal differences may be the cause of the increased insomnia risk in women.
  Fact 1.2: Dr Browning names menstruation, pregnancy and menopause as examples of hormonal differences.
  Fact 1.3: Dr Browning says hormonal differences such as menstruation, pregnancy and menopause may cause this increased risk in women.

Retrieved items:
<item 1>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“People who have insomnia are desperate. I treat patients who are suicidal, who have quit their jobs, whose lives are completely on hold because of their
sleeping issues. And they’re vulnerable. My clients have spent hundreds and hundreds of pounds on things that made no difference. If people are being mis-sold
expensive things and they’re being preyed upon, that’s a huge problem.”

Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications, teas, supplements, weighted blankets, and eye masks. “While supplements, exercise, and sleep hygiene provided some relief, the
true breakthrough came from addressing the root causes of my insomnia through therapy,” she tells me.
</item 1>

<item 2>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
“One of the most important aspects of a healthy sleep schedule is waking up around the same time each morning, so try to limit yourself to a one-hour window if
you need to adjust when you get up in the morning and when you go to sleep at night. One hour sounds like an insignificant amount of time, but it can completely
throw our circadian rhythm, or more simply known as internal clock, off balance,” he says.

Despite this, a spokesperson for Sleep School, a science-based sleep education and therapy service, told The Independent that it is “impartial” on the extent to
which the clocks changing impacts sleep, arguing that the important factor is the duration of time asleep and a consistent schedule.

Fewer depressive episodes

A Danish study from 2017 of more than 185,000 people over 15 years found that the transition from summer time to standard time was associated with an 11 per
cent rise in depressive episodes. Conversely, the switch from winter to summer time was found to have no effect.
</item 2>

<item 3>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The series was inspired by creator Natalie Walter’s own experience with insomnia, and her late-night Facebook chats with Steve Burge, her co-writer on Still Up.
“As part of the research for [the show] we visited a sleep clinic, which was fascinating,” she tells me. “Some evolutionary biologists think the reason humans
have different sleep patterns is that during our early history, a few members of the tribe would always be awake and vigilant during the night, and so could
protect everyone else from predators. Not that I’d be much use fighting off a woolly mammoth.”

Just like depression or anxiety, insomnia can often be triggered by something. Whether it’s a life event as monumental as becoming a new parent, or something as
commonplace as jet lag, there are any number of things that can prompt a bout of sleeplessness. The issue, explains sleep therapist Dr Lindsay Browning, is that
insomnia can often remain long after the trigger has gone.
</item 3>

<item 4>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The symptoms of insomnia can last a long time; a longitudinal study conducted in the UK found that 69 per cent of respondents affected by the sleep disorder at
the start of the study still had insomnia 12 months later.

This may go to show how complex treating insomnia can be. Explaining the appeal of quick fixes, Dr Browning says, “In the short term you’re changing something,
so [they] can often [work] like a placebo. If hunger is keeping you awake, eating before bed will help. If your mattress is uncomfortable, sleeping on the floor
will help.” But the problem isn’t just that these methods don’t necessarily work – it’s that they could be used to exploit vulnerable insomniacs.

“I’m a sleep therapist, my doctorate is in insomnia and it’s very, very frustrating,” she says. “People who have insomnia are desperate.
</item 4>

<item 5>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Long-time insomniac and video producer Preshita says she has tried many tricks over the years – “from drinking chamomile tea to counting backwards from 1,000”.
After much experimenting, she has found that taking a shower before bed and keeping her bedroom clutter-free were the most effective methods. “Someone
recommended masturbating, and that was another thing that worked,” she adds.

‘Still Up’ follows Lisa (Antonia Thomas) and Danny (Craig Roberts) as best friends who stay up late talking to each other because they can’t sleep (Courtesy of
Apple)

Millions of people struggle with insomnia in the UK alone. Studies have found that women are 40 per cent more likely to be affected by insomnia than men. “This
sex difference in insomnia emerges after puberty, suggesting that hormonal differences such as menstruation, pregnancy and menopause may be the cause of this
increased risk in women,” Dr Browning says.
</item 5>

<item 6>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Lavender oil. Eating kiwis. Masturbation. Everyone has their own “hacks” when it comes to falling asleep in 2023; the days of counting sheep are long gone.
Jennifer Aniston’s top tip for battling insomnia is to “sleep with my phone at least five feet away from me”, while Martha Stewart recommends focusing on your
breathing.
</item 6>

<item 7>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
In her 1973 feminist novel Fear of Flying, author Erica Jong notes that, sometimes, the only way to overcome sleeplessness is by pretending you don’t care about
nodding off. “Then sleep became piqued, like a rejected lover, and crept up to try to seduce you.”

In a world where human sleep patterns have increasingly been plunged into chaos, people are having to turn to ever more creative – and often bizarre – means of
getting their 40 winks in. Today, the internet is littered with tips and tricks that promise a sound sleep, but not all of them are effective and none are
foolproof. (Kiwis, by the way, contain serotonin and antioxidants, which is why it’s thought they might have sleep-promoting benefits.) The result is a society
with a growing problem on its hands: persistent sleeplessness – otherwise known as insomnia.
</item 7>

<item 8>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Globally, it is estimated that around 10 per cent of adults meet the criteria for insomnia; the sleep disorder is characterised either by difficulty falling
asleep or staying asleep to such an extent that it disturbs your ability to function during the day.

Insomnia lies at the heart of the new Apple TV+ comedy series Still Up. The show follows Lisa (Antonia Thomas) and Danny (Craig Roberts), two best friends who
stay up late talking to each other because they can’t sleep. Lit by the blue glow emanating from their electronic devices, Lisa and Danny form a deep connection
over their nocturnal calls, with their friendship gradually drifting into something more. At one point, Lisa checks herself into a sleep clinic in London,
telling Danny she’s willing to “try anything” at this point – a sentiment the sleepless know all too well.
</item 8>

<item 9>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“At this point, you’re no longer worried about the traumatic incident; you’re now worried about not being able to sleep.” It’s a self-fulfilling prophecy.

Social media consultant and mother of two, Amy Fieldhouse-Downes, started suffering from insomnia after the birth of her second son in 2019. She believes the
affliction was rooted in anxiety. “If you Google how to go to sleep, I’ve probably tried everything on there,” she says. “Going to bed earlier, going to bed
later, meditations that would get cut off by an advert, lavender products, sleep cream, eye masks, Sominex sleep tablets. And I even used it as an excuse to
have a few glasses of wine.” It wasn’t until she tried sleeping with specialised earplugs that she was finally able to get some rest.
</item 9>

<item 10>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Research shows that even a small reduction in sleep has been shown to impact your immune system, however.

Making the move from daylight saving time interferes with your natural circadian rhythm, the 24-hour biological cycle that is influenced by morning light and
evening darkness, however.

“It’s hard to understate the importance of a good night’s sleep,” says Giulia Guerrini, lead pharmacist of digital pharmacy Medino. “Having too much or too
little sleep can have adverse side effects, so you should aim to get between seven and eight hours of sleep as consistently as possible.”

Never underestimate the value of a good night’s sleep (Getty)

“One of the biggest disruptors to our sleep is when UK clocks go forward one hour at 1am on the last Sunday in March and go back one hour at 2am on the last
Sunday in October, which is rapidly approaching on 30 October.
</item 10>
=== END ITEM id=5d0fb9a9c69a58c0 ===

=== ITEM id=34df5c5926dd6f7c ===
Gold sentences and facts:
Sentence 0: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 0.0: Google unveiled new hardware.
  Fact 0.1: This week was Google's annual hardware event.
  Fact 0.2: Google is a search and consumer tech giant.
  Fact 0.3: At the hardware event, Google showed off what Google had been working on.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The last roundup reported how Google spent $26.3 billion in 2021 making Google the default search engine across platforms.
  Fact 1.1: The last roundup reported how Google tried to have Chrome preinstalled on iPhones.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google "siphons off" news publishers' readers.
  Fact 2.3: The case argues that Google "siphons off" news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 5>

<item 6>
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
</item 6>

<item 7>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 7>

<item 8>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 10>
=== END ITEM id=34df5c5926dd6f7c ===
