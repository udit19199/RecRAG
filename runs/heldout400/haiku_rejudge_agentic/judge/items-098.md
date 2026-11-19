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

=== ITEM id=221e1df0484f1a0d ===
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
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
The performance is fast, the recharging is fast. It even looks fast. It has a bright, 120-Hz AMOLED screen, great speakers, and surprisingly good battery life
for all that. Our main gripe is the IP64 water- and dust-resistance rating, which is not nearly as good as other flagship phones.

Ugreen's charger is, as the name suggests, a 145-watt charger with a 25,000-mAh battery. It's surprisingly compact for the power it provides, although at 1.1
pounds, it's definitely not ultralight. There are two USB-C ports and one USB-A port. What sets the Ugreen apart is that you can actually draw 145 watts while
charging. That works out to one USB-C port at 100W and the other at 45W. Very few other batteries we've tested are capable of that feat.

Photograph: Anker

The Anker Nano is one of our favorite portable power banks, particularly for phones.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -PUBLISHED_BY-> WIRED
- These Are Our Favorite Walmart Cyber Monday Deals -PUBLISHED_BY-> WIRED
- Wired opinion article -PUBLISHED_BY-> WIRED
- 14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds -PUBLISHED_BY-> WIRED
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -PUBLISHED_BY-> WIRED

Entities: WIRED, Ugreen, Ugreen charger, Anker Nano, 78 Absolute Best October Prime Day Deals (2023), Anker
</item 1>

<item 2>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Granted, it’s bulkier than buying one or two specialized adapter plugs, but if you’ve got a multi-continent itinerary, it’s a game-changer. What I like about
this one from Epicka is that it comes with a spare fuse in case things go sideways with voltage.

The only caveat is that, although it says “universal,” it’s technically only the four most common types of plugs. That’ll get you by in most countries, but it’s
not a guarantee in places like Brazil, South Africa, or India. Even so, I’ll take this over price gouging at airport electronics shops or having to take time
out of my schedule to visit a local hardware store.

Lovely lights

Brandon Widder, senior commerce editor

I’m a big fan of ambient lighting, even when I’m 50 miles from the nearest outlet.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Brazil
- Here’s how Rainforest, a budding Stripe rival, aims to win over software companies -MENTIONS-> Brazil
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> Brazil
- If your startup doesn’t seem impossible at first, it’s not hard enough -MENTIONS-> Brazil
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Brazil

Entities: Brazil, India, South Africa, Black Friday, Some of our favorite devices are on sale for Black Friday., Brandon Widder, Epicka, ambient lighting, The
Verge
</item 2>

<item 3>
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

Graph facts:
- The best Black Friday deals you can get for under $25 -MENTIONS-> Black Friday
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> Black Friday
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> Black Friday
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Black Friday
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Black Friday

Entities: Black Friday, Amazon, Some of our favorite devices are on sale for Black Friday., Sarah Jeong, Anker 321 Power Strip, The Verge
</item 3>

<item 4>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Older buildings especially suffer from what I can only describe as loose socket syndrome, where those very convenient modern boxy socket extenders with five
different USB and USB-C charging ports simply cannot stay in place and fall right out of the wall because they’re too heavy. After one (totally pleasant)
vacation where I had to charge my phone, watch, AirPods, and laptop in a weird corner of my room with the plug-in charging hub propped up on a strategically
balanced mountain of books and sham pillows, I bought this Anker combination extension cord / power strip. It’s not a full power strip — just a cube with a few
sockets along with USB and USB-C charging ports at the end of a five-foot cable. I’ve brought it on a few trips since then. It takes up extra space in my
suitcase but each time has left me feeling vindicated about the purchase.

The three prongs at the end of the cable are static, rather than folding flat for easy packing.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, Anker, Black Friday, Some of our favorite devices are on sale for Black Friday., Anker combination extension cord / power strip, The Verge
</item 4>

<item 5>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.

Graph facts:
- The Ray-Ban Meta smart glasses actually make the future look cool -MENTIONS-> Becca Farsace
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Black Friday
- The best Black Friday deals you can get for under $25 -MENTIONS-> Black Friday
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> Black Friday
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> Black Friday

Entities: Becca Farsace, Black Friday, Some of our favorite devices are on sale for Black Friday., MPOWERD, Luci lanterns, string lights, headlamp, Memory card
holder and reader, dongles, The Verge
</item 5>

<item 6>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Flip over to the back, and you'll find two USB-A ports, two USB-C ports, and three AC outlets. It's a compact and tidy charger, and our unit has been going
strong for more than two years. It fluctuates in price very often, and it's hit as low as $60 in the past, but we still think this is a good deal.

This power bank is one of our favorites if you're looking for an upgrade. It has an absolutely gargantuan 24,000-mAh capacity, and it can be charged up in an
hour. It can output up to 140W, making it powerful enough to charge laptops and tablets, as well as charging your phone super quick.

This is a great charger if you travel often with an iPhone and Apple Watch. Instead of packing two separate chargers, the Duo offers the luxury of both all
within a compact, foldable form factor.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, iPad, Black Friday deals, Apple Watch, WIRED, phone, power bank, tablets, laptops, iPhone, Duo, 54 Best Apple Black Friday Deals (2023):
iPad, Apple Watch, AirPods, charger
</item 6>

<item 7>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Black Friday and Cyber Monday may be in the past, but some sales are still hanging around (although maybe not for long). For more leftover deals we recommend
across all categories, be sure to check out the rest of the Cyber Monday still available here.

Every month or so, we like to ask our staff about their favorite stuff — whether it’s pet toys, travel aids, kitchen gadgets, or straightforward tech. And the
results are usually very different, very interesting, and a lot of fun.

In celebration of the annual post-Thanksgiving sales, we looked through some of our recent “favorites” articles and found deals on a lot of the tech, kitchen
tools, travel aids, and pet toys we like. We thought we’d list a few in case you’ve read about them in the past and thought, “Well, that sort of sounds good,
but it’s a bit pricey.”

Graph facts:
- Check Point sees 'fantastic' year ahead even as rival report disappointing billings -MENTIONS-> tech
- Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle -MENTIONS-> tech
- Shop 200+ Walmart doorbuster deals for Black Friday -MENTIONS-> tech
- Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne -MENTIONS-> tech
- Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more -MENTIONS-> tech

Entities: tech, Black Friday, Cyber Monday, kitchen tools, travel aids, pet toys, Some of our favorite devices are on sale for Black Friday., The Verge
</item 7>

<item 8>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
There's even a built-in display to monitor stats like temperature and remaining battery percentage.

This heavy metal brick is a good option for charging laptops and smaller gadgets. It packs 20,000 mAh and supports a wide variety of fast charging standards.
There is one USB-C PD port rated at 65 watts, two USB-A QC ports at 30 watts apiece, and a micro-USB input (though you are best using the USB-C to recharge it).
We haven't seen it go on sale very often.

This travel kit from ESR works well if you want something that can prop your iPhone in portrait or landscape orientation. It can also display your Apple Watch
in Nightstand mode, so you can peek and see how long before you must get out of bed. The main body folds open with a MagSafe charging pad for your iPhone and a
slot behind for your AirPods. We have seen this dip a little lower, but this is still a solid deal.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, Apple Watch, WIRED, October Prime Day, iPhone, heavy metal brick, MagSafe charging pad, 78 Absolute Best October Prime Day Deals (2023), ESR,
travel kit
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

Graph facts:
- The best Black Friday deals you can get for under $25 -MENTIONS-> Epicka universal travel adapter
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Korea
- 9 great movies to watch on Tubi for free right now -MENTIONS-> Korea
- 10 scorching-hot discoveries made about the sun in 2023 -MENTIONS-> Korea
- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings -MENTIONS-> Korea

Entities: Epicka universal travel adapter, Korea, Black Friday, Amazon, Victoria Song, Some of our favorite devices are on sale for Black Friday., The Verge
</item 9>

<item 10>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
A car-friendly charger

Sean Hollister, senior editor

Maybe someday Apple will realize that it wasn’t the brightest idea to artificially limit the reach of its MagSafe charging ecosystem and you’ll be able to plop
your phone on a magical minimalist disc that charges it at high speeds. In the meanwhile, a standard Qi charger is about the best you’ll get — and the bulky but
practical iOttie Easy One Touch Qi does it with the satisfying snap of springs. When you push your phone into its waiting jaws, it depresses a button that
causes those jaws to firmly snap closed on either side of your device. When you want to remove it, you pinch a pair of levers with your finger and thumb to
release as you grab your slab. It’s wide enough to fit practically anything on the market, save an opened Samsung Z Fold. I’ve used one for years with Android
and Apple phones alike, including newer MagSafe handsets.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, Apple, Samsung, Black Friday, Some of our favorite devices are on sale for Black Friday., Sean Hollister, MagSafe charging ecosystem,
standard Qi charger, iOttie Easy One Touch Qi, Samsung Z Fold, The Verge
</item 10>
=== END ITEM id=221e1df0484f1a0d ===

=== ITEM id=2ff2d60538c7a0d2 ===
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
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.

Graph facts:
- McRae reveals hidden message he carried under collar of his grand final shirt -MENTIONS-> Adams
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Adams
- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat -MENTIONS-> Adams
- NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable -MENTIONS-> Adams
- Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Adams

Entities: Adams, Ghana, Gregg Berhalter, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, Johnny
Cardoso, Internacional, Lennard Maloney, Heidenheim, October friendlies, sports hernia, September camp, October squad call-up, USMNT, Sporting News
</item 1>

<item 2>
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

Graph facts:
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana

Entities: Oman, Ghana, Gregg Berhalter, Celtic, AC Milan, October friendlies against Germany and Ghana, Friendlies against Uzbekistan and Oman, Uzbekistan,
Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, Ethan Horvath, Nott. Forest, USA squad for October friendlies 2023: USMNT roster for friendly matches
vs Germany, Ghana
</item 2>

<item 3>
Graph query row:
publisher: Sporting News
title: USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana
publication_date: 2023-10-04T22:42:00+00:00
</item 3>

<item 4>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Juventus, Gregg Berhalter, Union Berlin, AC Milan, Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, USA squad for
October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, Tyler Adams, October camp, Gio Reyna
</item 4>

<item 5>
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

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Gregg Berhalter, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, leg fracture,
Gio Reyna, summer Nations League duty, domestic violence investigation of Berhalter, domestic violence, USMNT, Sporting News, Germany, 2022 World Cup, Borussia
Dortmund
</item 5>

<item 6>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
“I was able to put lessons from that time into practice, develop and eventually return to the Socceroos much more prepared for the role. It’s been a privilege
to be part of the national team again with the benefit of that experience, and I’m immensely proud of what this group has been able to achieve in the last four
years.

“In my opinion, there’s no greater honour than representing your nation on the international stage. I look at the young players coming through the Socceroos
system now full of pride in the work they’ve put in to get where they are, and the journey that’s ahead of them.

“I’ve always loved being a part of the Socceroos culture and my desire to see Australia fulfil its footballing potential is still driving me to take this group
to further success.”

Australia starts the road to north America in Melbourne at AAMI Park on Thursday night, before travelling to the Middle East to play Palestine next Wednesday
(1.00am AEDT).

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Australia
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Australia
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Australia
- Israel's defense undone by reliance on technology -MENTIONS-> Australia
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Australia

Entities: Australia, ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, Arnie, AAMI Park, Australia
starts the road to North America, Play Palestine, Melbourne, Middle East, North America, The Roar, Palestine
</item 6>

<item 7>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Wolfsburg
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Juventus
- Wolfsburg -PARTICIPATED_IN-> 2015-16 deciding group match
- Manchester United -COMPETES_WITH-> Wolfsburg
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Juventus

Entities: Wolfsburg, Juventus, Fulham, New England Revolution, Celtic, AC Milan, Ghana, Malik Tillman, PSV Eindhoven, USA squad for October friendlies 2023:
USMNT roster for friendly matches vs Germany, Ghana, friendly matches vs Germany, Ghana, USA, Cameron Carter-Vickers, Sergino Dest, DeJuan Jones
</item 7>

<item 8>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Bournemouth, Leeds United, Celtic, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs
Germany, Ghana, Cameron Carter-Vickers, Zack Steffen, Sean Johnson, Josh Sargent, Daryl Dike, leg fracture, Achilles tear, October friendlies 2023
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

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Dortmund, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies 2023, Berhalter,
Reyna, USMNT, Sporting News, Germany, Zoom
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.

Graph facts:
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Gregg Berhalter
- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat -MENTIONS-> Gregg Berhalter
- What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager -MENTIONS-> Gregg Berhalter
- USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah -MENTIONS-> Gregg Berhalter
- Gregg Berhalter -SAID-> Antonee Robinson

Entities: Gregg Berhalter, Connecticut, 2024 Copa America, U.S. men's national team, 2026 FIFA World Cup, Euro 2024 qualifying, Ghana, friendly against Ghana,
GEODIS Park, Nashville, World Cup tournament in 2006, World Cup tournament in 2010, World Cup tournament in 2014, USA squad for October friendlies 2023: USMNT
roster for friendly matches vs Germany, Ghana, friendly against Germany
</item 10>
=== END ITEM id=2ff2d60538c7a0d2 ===

=== ITEM id=b3e453432e30a78d ===
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
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Juventus, Gregg Berhalter, Union Berlin, AC Milan, Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, USA squad for
October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, Tyler Adams, October camp, Gio Reyna
</item 1>

<item 2>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Dortmund, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies 2023, Berhalter,
Reyna, USMNT, Sporting News, Germany, Zoom
</item 2>

<item 3>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.

Graph facts:
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> World Cup
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> World Cup
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> World Cup
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> World Cup
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> World Cup

Entities: World Cup, Gregg Berhalter, Tyler Adams, Gio Reyna, USMNT, 2026 World Cup, Gio Reyna looks good in start for USMNT vs. Germany, but those involved in
the finish are punished in 3-1 defeat, USMNT vs. Germany, Sporting News, Germany, North America
</item 3>

<item 4>
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

Graph facts:
- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana -MENTIONS-> Johnny Cardoso
- Kevin Paredes -WORKS_FOR-> Wolfsburg
- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana -MENTIONS-> Kevin Paredes
- Kevin Paredes -PART_OF-> USA
- Johnny Cardoso -WORKS_FOR-> Internacional

Entities: Kevin Paredes, Johnny Cardoso, Luca de la Torre, Lennard Maloney, USMNT, Pulisic, Sporting News, Germany, USA vs Germany player ratings: USMNT
defensive errors blight positive performance by Pulisic, Weah, Weah, player ratings, defensive errors, USA vs Germany
</item 4>

<item 5>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Bournemouth, Leeds United, Celtic, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs
Germany, Ghana, Cameron Carter-Vickers, Zack Steffen, Sean Johnson, Josh Sargent, Daryl Dike, leg fracture, Achilles tear, October friendlies 2023
</item 5>

<item 6>
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

Graph facts:
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana

Entities: Oman, Ghana, Gregg Berhalter, Celtic, AC Milan, October friendlies against Germany and Ghana, Friendlies against Uzbekistan and Oman, Uzbekistan,
Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, Ethan Horvath, Nott. Forest, USA squad for October friendlies 2023: USMNT roster for friendly matches
vs Germany, Ghana
</item 6>

<item 7>
Graph query row:
person: Gary Glover
report: ASX set to drop as Wall Street’s September slump deepens
published_on: 2023-09-26T19:11:30+00:00
passage: ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
ETF provider Betashares, which manages $30 billion in funds, reached an agreement to acquire Bendigo and Adelaide Bank’s superannuation business, in its first
venture into the superannuation sector. Betashares said it was part of a longer-term strategy to expand the business into the broader financial sector. Shares
in Bendigo increased 0.6 per cent on the news. REITS (up 0.4 per cent) was the strongest sector on the index as Goodman added 0.5 per cent and Dexus climbed 2.8
per cent. The laggards Casino operator Star Entertainment Group’s shares hit an all-time low of 60¢ after it raised $565 million. They closed the session 16 per
cent weaker at 63¢. Star, which raised $800 million in February, has had to return to the market for fresh funding and is hoping to raise $750 million at a
share price of 60¢ a share.
next_passage: None
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

Graph facts:
- McRae reveals hidden message he carried under collar of his grand final shirt -MENTIONS-> Adams
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Adams
- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat -MENTIONS-> Adams
- NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable -MENTIONS-> Adams
- Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Adams

Entities: Adams, Ghana, Gregg Berhalter, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, Johnny
Cardoso, Internacional, Lennard Maloney, Heidenheim, October friendlies, sports hernia, September camp, October squad call-up, USMNT, Sporting News
</item 8>

<item 9>
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

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Gregg Berhalter, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, leg fracture,
Gio Reyna, summer Nations League duty, domestic violence investigation of Berhalter, domestic violence, USMNT, Sporting News, Germany, 2022 World Cup, Borussia
Dortmund
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Wolfsburg
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Juventus
- Wolfsburg -PARTICIPATED_IN-> 2015-16 deciding group match
- Manchester United -COMPETES_WITH-> Wolfsburg
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Juventus

Entities: Wolfsburg, Juventus, Fulham, New England Revolution, Celtic, AC Milan, Ghana, Malik Tillman, PSV Eindhoven, USA squad for October friendlies 2023:
USMNT roster for friendly matches vs Germany, Ghana, friendly matches vs Germany, Ghana, USA, Cameron Carter-Vickers, Sergino Dest, DeJuan Jones
</item 10>
=== END ITEM id=b3e453432e30a78d ===
