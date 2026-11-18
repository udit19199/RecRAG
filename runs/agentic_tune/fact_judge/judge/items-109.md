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

=== ITEM id=914064152f0f6128 ===
Gold sentences and facts:
Sentence 0: Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest, Sophie.
  Fact 0.0: Zazie Beetz will be back as Sophie.
  Fact 0.1: Sophie is Arthur's neighbor.
  Fact 0.2: Sophie is Arthur's entirely unwitting love interest.
Sentence 1: But then, there’s MJ as well who has made a comeback from the first installment.
  Fact 1.0: MJ is also present in the story.
  Fact 1.1: MJ has made a comeback from the first installment.

Retrieved items:
<item 1>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Mary’s gameplay missions end up increasing the overall quality of Spider-Man 2. All things considered, it is great to see Insomniac improving on this aspect
instead of removing it completely.

ADVERTISEMENT Article continues below this ad

Marvel’s Spider-Man 2 is now available on PlayStation 5 worldwide.

Watch This Story: The Console Wars Are Heating Head-to-Head: Xbox and PlayStation Fans Battle Over Which Custom Controller is Better, Starfield or Spider-Man 2?

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Xbox
- Xbox -MENTIONS-> mobile market
- Sarah Michelle Gellar hits out at Roblox after her son’s account gets hacked -MENTIONS-> Xbox
- Microsoft Activision: What does deal mean for gamers? -MENTIONS-> Xbox
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Xbox

Entities: Xbox, Insomniac, Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Starfield, Essentially
Sports, PlayStation, PlayStation 5
</item 1>

<item 2>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
But how?

Compared to the lack of interactivity in 2018’s Spider-Man, MJ is now capable of tackling the enemies around her. In the first title, all she could do was to
move from one point to the other. Now, she’s able to do more than that – make pretty good and lethal moves. To be precise, the devs have given MJ a little
firepower, if not in too much capacity. Now, she’s got weapons instead of being clueless in moments of being vulnerable against armed enemies. Yes, she hasn’t
got any superpowers, but she’s got other means to help herself out.

Due to such involvement, players have now got a reason to be more invested into a given mission focusing on MJ. It wasn’t the case at all taking into
consideration her previous iteration in Insomniac’s original game. Mary Jane can now handle the circumstances in her own manner, thus breaking free from the
linear approach of just trying to hide and escape.

Graph facts:
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Insomniac
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Spider-Man, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially
Sports
</item 2>

<item 3>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
There’s a visible growth in her arc especially considering her profession as a journalist. Yes, she’s Peter’s love interest, but that’s not everything she’s
there for. Players can easily see it happening throughout the game’s duration.

The verdict?

ADVERTISEMENT Article continues below this ad

It’s obvious to expect that a lot of fans who didn’t like MJ’s missions in the first installment won’t like her this time as well. But considering how she was
before and how she’s in Spider-Man 2, the overall change is quite praiseworthy. Things are different this time, thankfully.

MJ’s missions are now having some actual meaning attached to them rather than acting like fillers. Players who want to have a few interesting moments aside from
the main quests featuring Miles and Peter can have a sense of finishing well-designed stealth quests. Mary’s gameplay missions end up increasing the overall
quality of Spider-Man 2.

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Xbox
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Insomniac
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Marvel’s Spider-Man 2
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> The Daily Bugle
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Spider-Man

Entities: Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, journalist, Peter, Essentially Sports, Spider-Man 2, Miles
</item 3>

<item 4>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
(If you can get past the Frankenstein-ification of the boroughs, that is. RIP Chelsea.)

There are hints of greater thematic aspirations within Spider-Man 2. Questions about mass incarceration, recidivism, and reintegration into society for formerly
incarcerated people; worries about the surveillance state and what we give up when we engage with modern technology. These things appear, however fleetingly, in
Spider-Man 2, but there isn’t enough time to elaborate on any single one in the unrelenting march of set-piece after glorious set-piece.

But I’m glad they’re there, anyway. Because if we’re going to retell a story, we ought to retell it anew. Canons were made to be broken, and Spider-Man 2 swings
in that direction. With a sequel teed up by the game’s final act, I’m eager to see where Insomniac goes with that momentum. Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct.

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Insomniac
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Marvel’s Spider-Man 2
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Marvel, Spider-Man 2 toys with Marvel history to be as much fun as possible, Chelsea, the Bronx, mass incarceration,
recidivism, reintegration into society, surveillance state, modern technology, release of Marvel’s Spider-Man 2, Polygon
</item 4>

<item 5>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Insomniac’s team, Peter, Essentially
Sports, Miles
</item 5>

<item 6>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
MJ in Spider-Man 2 vs. MJ from Insomniac’s 2018 installment

While there are similarities between the character’s iterations in two titles, a lot of drastic differences are also present. Speaking of similar aspects, the
first game had a mission where Mary gets to use a taser to shock the guards. Well there’s no taser in Spider-Man 2, she’s got a stun gun now.

Then, there are moments of hiding and sneaking past the enemies. In the original game, there were missions like the one in Grand Central Station where MJ had to
stay hidden from a number of Mr. Negative’s demons to defuse a bomb. Similarly, in Spider-Man 2, MJ is seen hiding from the hunters during the ‘Wake Up’ quest.
But what’s actually changed?

Well, the very difference lies in the lack of MJ’s abilities in the first game and how she’s evolved in the second.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Insomniac’s 2018 installment, Grand Central
Station, Mr. Negative, Mr. Negative’s demons, hunters, Wake Up quest, taser, stun gun, bomb, Mary Jane, Essentially Sports
</item 6>

<item 7>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
One can even call her the third unofficial superhero in the latest PS5 exclusive.

A refreshing take on Mary Jane’s stealth missions makes her a strong character in Marvel’s Spider-Man 2

ADVERTISEMENT Article continues below this ad

In the first title, stepping into MJ’s shoes was a great concept, but the devs somehow couldn’t execute the plan flawlessly. As a result, missions featuring her
as a playable character were heavily criticized by most of the fans. There was nothing too special about those quests whatsoever.

Now, looking at Mary Jane in Marvel’s Spider-Man 2, there’s a very clear difference one can make out. In short, it feels comparing night and day. Being a
journalist at The Daily Bugle, her character is bound to go places in order to investigate mysteries. The first game had such situations, but the sequel
improved on them. It gives MJ a lot more strength instead of being someone who always needs to be rescued in the last moment. But how?

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, The Daily Bugle, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially Sports, PS5
</item 7>

<item 8>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
With the recent release of Marvel’s Spider-Man 2, Insomniac has witnessed a huge appreciation from fans for delivering a memorable sequel. Not to mention the
epic portrayal of Peter Parker and Miles Morales fighting against some of the strongest villains. But then, there’s MJ as well who has made a comeback from the
first installment. And it’s her arc that has actually received an impressive elevation this time. Especially when considering the poorly received missions
featuring Mary Jane in the original game from 2018.

America’s Favorite Video Today

So what’s actually changed in 2023 that has created a vast difference? There are certain key reasons making Mary Jane in Spider-Man 2 a treat to watch. Not just
from the perspective of her relationship with Peter, but her own missions showcasing a different side this time. One can even call her the third unofficial
superhero in the latest PS5 exclusive.

Graph facts:
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Insomniac
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially Sports, Peter
Parker, Miles Morales, PS5
</item 8>

<item 9>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Peter’s plot involves a retelling of the Venom origin story, with beats that will be recognizable to those even passingly familiar with the Spider-Man mythos.
If Spider-Man 2 is a game of retellings, Peter’s story clings most closely to the canon. If you’ve seen Sam Raimi’s Spider-Man 3, you’ll know the gist of what
to expect when the symbiote finds its way to Peter. It isn’t until the late game that Peter’s story starts pushing more strongly against our canonical memories
and into something more unique to Insomniac.

Miles’ story takes more liberties. Hailey Cooper, Miles’ love interest from his solo game, returns here, taking more of an MJ-esque role and carving out her own
identity separate from Miles’. American Sign Language is featured heavily throughout Miles’ missions because of Hailey’s presence, feeling as natural and
integral to the story as anything from the canon.

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Insomniac
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Peter
- The best video games of the year so far -MENTIONS-> Peter
- Thanksgiving is Spider-Man’s holiday -MENTIONS-> Peter

Entities: Insomniac, Peter, symbiote, Spider-Man 2, Miles, Spider-Man 3, Venom origin story, Spider-Man mythos, Hailey Cooper, MJ, American Sign Language,
Spider-Man 2 toys with Marvel history to be as much fun as possible, Polygon, Sam Raimi
</item 9>

<item 10>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Coming back to the ‘Wake Up’ mission, she’s seen sneaking past the hunters but she’s also capable of performing Takedowns on them. It wasn’t the case in the
previous title. Her missions have become better now with the devs actually spending time on giving her a chance to fight.

Coming back to the stun gun part, MJ is pretty skilled in using it, during missions like ‘Good Men’, thanks to her training with Sable. Also, now she’s got the
option to even run away and hide when spotted by enemies. The playability factor around her has increased a lot compared to being just an observer, for the most
part, in the first game. That’s the biggest and most visible difference.

Mary Jane and her missions in Spider-Man 2 allow the story to proceed without causing a problem for the central protagonists

The best thing about going through the quests showcasing Mary Jane is that they never try to overpower the ongoing story itself.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Wake Up, Good Men, Sable, Essentially
Sports
</item 10>
=== END ITEM id=914064152f0f6128 ===

=== ITEM id=be1fdd1b8508d78a ===
Gold sentences and facts:
Sentence 0: “Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022.
  Fact 0.0: The speaker added that the team can hopefully go one better this year.
  Fact 0.1: The speaker recalled the 2-0 defeat by Canada in 2022.
Sentence 1: After a winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season.
  Fact 1.0: The Swans had a winless campaign in 2022.
  Fact 1.1: The Swans are led by Chloe Molloy.
  Fact 1.2: Chloe Molloy is an ex-Collingwood player.
  Fact 1.3: The Swans have been competitive this season.

Retrieved items:
<item 1>
Graph query row:
article: ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final
connections: [{'entity': 'The Age', 'relationship': 'PUBLISHED_BY'}, {'entity': 'Getty Images', 'relationship': 'MENTIONS'}, {'entity': 'Russia',
'relationship': 'MENTIONS'}, {'entity': 'Canada', 'relationship': 'MENTIONS'}, {'entity': 'Australia', 'relationship': 'MENTIONS'}, {'entity': 'Novak Djokovic',
'relationship': 'MENTIONS'}, {'entity': 'Carlos Alcaraz', 'relationship': 'MENTIONS'}, {'entity': 'Jannik Sinner', 'relationship': 'MENTIONS'}, {'entity':
'Serbia', 'relationship': 'MENTIONS'}, {'entity': 'Finland', 'relationship': 'MENTIONS'}, {'entity': 'Darren Cahill', 'relationship': 'MENTIONS'}, {'entity':
'Daniil Medvedev', 'relationship': 'MENTIONS'}, {'entity': 'Lorenzo Sonego', 'relationship': 'MENTIONS'}, {'entity': 'Max Purcell', 'relationship': 'MENTIONS'},
{'entity': 'Matt Ebden', 'relationship': 'MENTIONS'}, {'entity': 'Italy', 'relationship': 'MENTIONS'}, {'entity': 'Davis Cup semi-final triumph over Serbia',
'relationship': 'MENTIONS'}, {'entity': '2022 Wimbledon championships', 'relationship': 'MENTIONS'}, {'entity': 'Popyrin', 'relationship': 'MENTIONS'},
{'entity': 'Filippo Volandri', 'relationship': 'MENTIONS'}, {'entity': 'Lorenzo Musetti', 'relationship': 'MENTIONS'}, {'entity': 'Miomir Kecmanovic',
'relationship': 'MENTIONS'}, {'entity': 'Matteo Arnaldi', 'relationship': 'MENTIONS'}, {'entity': 'Arthurs', 'relationship': 'MENTIONS'}, {'entity': 'Miami
Masters', 'relationship': 'MENTIONS'}, {'entity': 'ATP Tour', 'relationship': 'MENTIONS'}, {'entity': 'leg issue', 'relationship': 'MENTIONS'}, {'entity':
'Alexei Popyrin', 'relationship': 'MENTIONS'}, {'entity': 'Musetti', 'relationship': 'MENTIONS'}, {'entity': 'Sonego', 'relationship': 'MENTIONS'}, {'entity':
'junior French Open', 'relationship': 'MENTIONS'}, {'entity': 'Davis Cup win', 'relationship': 'MENTIONS'}, {'entity': 'Finn Otto Virtanen', 'relationship':
'MENTIONS'}, {'entity': 'playing for your country', 'relationship': 'MENTIONS'}, {'entity': 'Davis Cup final', 'relationship': 'MENTIONS'}, {'entity': 'Davis
Cup round-robin matches', 'relationship': 'MENTIONS'}, {'entity': 'Davis Cup knockout stage', 'relationship': 'MENTIONS'}, {'entity': 'De Minaur',
'relationship': 'MENTIONS'}, {'entity': 'Felix Auger-Aliassime', 'relationship': 'MENTIONS'}, {'entity': 'Denis Shapovalov', 'relationship': 'MENTIONS'},
{'entity': '9GEM', 'relationship': 'MENTIONS'}, {'entity': 'Italy team', 'relationship': 'MENTIONS'}, {'entity': 'Serbs', 'relationship': 'MENTIONS'},
{'entity': 'Sinner', 'relationship': 'MENTIONS'}, {'entity': 'Djokovic', 'relationship': 'MENTIONS'}, {'entity': 'singles rubber', 'relationship': 'MENTIONS'},
{'entity': 'doubles', 'relationship': 'MENTIONS'}, {'entity': 'ATP finals showdown', 'relationship': 'MENTIONS'}, {'entity': 'Kecmanovic', 'relationship':
'MENTIONS'}, {'entity': 'tie’s opening singles', 'relationship': 'MENTIONS'}, {'entity': 'third singles duel', 'relationship': 'MENTIONS'}, {'entity':
'Wimbledon final', 'relationship': 'MENTIONS'}, {'entity': 'Turin', 'relationship': 'MENTIONS'}, {'entity': 'Davis Cup', 'relationship': 'MENTIONS'}, {'entity':
'Palacio de deportes Martin Carpena', 'relationship': 'MENTIONS'}, {'entity': 'Lleyton Hewitt', 'relationship': 'MENTIONS'}, {'entity': 'Australian side',
'relationship': 'MENTIONS'}, {'entity': 'Italy tie', 'relationship': 'MENTIONS'}, {'entity': 'Malaga', 'relationship': 'MENTIONS'}, {'entity': 'Costa del Sol',
'relationship': 'MENTIONS'}, {'entity': 'Hewitt', 'relationship': 'MENTIONS'}, {'entity': 'Emil Ruusuvuori', 'relationship': 'MENTIONS'}, {'entity': 'Serbian
side', 'relationship': 'MENTIONS'}, {'entity': 'Italian team', 'relationship': 'MENTIONS'}, {'entity': 'Davis Cup semi-final', 'relationship': 'MENTIONS'},
{'entity': 'Otto Virtanen', 'relationship': 'MENTIONS'}, {'entity': 'Alex de Minaur', 'relationship': 'MENTIONS'}, {'entity': 'Jiri Lehecka', 'relationship':
'MENTIONS'}, {'entity': 'Czech Republic', 'relationship': 'MENTIONS'}, {'entity': 'quarter-final with the Czech Republic', 'relationship': 'MENTIONS'},
{'entity': '2-0 defeat by Canada', 'relationship': 'MENTIONS'}, {'entity': 'Thanasi Kokkinakis', 'relationship': 'MENTIONS'}, {'entity': 'Purcell',
'relationship': 'MENTIONS'}, {'entity': 'Jordan Thompson', 'relationship': 'MENTIONS'}, {'entity': 'Getty', 'relationship': 'MENTIONS'}]
passages: ['‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nThe second of Sinner’s
Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on serve, before besting the
world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice in the same season since
Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final. Credit: Getty However, if
Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good one, with 2022 Wimbledon
champions Max Purcell and Matt Ebden to be unleashed.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age,
2023-11-24T23:10:22+00:00)\nThe query is whether Italian captain Filippo Volandri will turn to Lorenzo Musetti, who battled a leg issue at the end of his
three-set loss to Serbia’s Miomir Kecmanovic, or one of Lorenzo Sonego or Matteo Arnaldi.\n\nPopyrin won his only ATP Tour clash with Musetti – who has been as
high as No.15 in the world – in a tight two-setter at last year’s Miami Masters, but the emerging Italian claimed three of their four showdowns at the next
level down. He is yet to play against Sonego, who Arthurs expects to play. “I’m assuming they will back in Popyrin because he played pretty well a couple of
nights ago, and with his firepower – he’s probably the key to the tie,” Arthurs said. “He’s showing all the signs of becoming a top-25 player in the next couple
of years.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nHis junior career
suggested he was going to be a very good player, and it’s unusual for someone to win the junior French Open and not go on to something big. Alexei Popyrin
celebrates with Australian captain Lleyton Hewitt after his Davis Cup win. Credit: Getty Images “These moments can be a big springboard for someone like Alexei.
He’s a confidence player, and this is a very big moment – whoever he plays, Musetti or Sonego. But he’s got to go out and embrace the moment, and he’s the type
of player who will enjoy it.\n\n“I’d love to give de Minaur a chance with his fighting abilities, but the way Sinner has played the last six months, you’d have
to say Sinner is the favourite.”', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age,
2023-11-24T23:10:22+00:00)\nLoading Hewitt animatedly embraced Popyrin after his straight-sets victory over 171st-ranked Finn Otto Virtanen, but Australia’s
Davis Cup captain would be even more thrilled if he could repeat the dose against Italy. Australia could also turn to an alternative No.2 singles option, with
Jordan Thompson and Purcell waiting in the wings. “It’s everybody’s dream to play for their country,” Popyrin said. “I looked up to Lleyton, looked up to all
the older Aussie legends. To be able to put on the green and gold, to be able to go out there and play for your country, no matter the format, if it’s this
Davis Cup or the old Davis Cup, you give your all.\n\n“It’s nerves, it’s pressure – you’re playing for everybody back home. It’s something that nothing will
ever compare to ... it really was the biggest moment of my career to get the first live rubber win, to get one foot into the final.', '‘Biggest win of my
career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nIt was something that I’ll never forget.” Loading
Australia, who lost last year’s final to Canada, has enjoyed a soft run on paper to this year’s decider, eliminating the Czech Republic and Finland – with
neither country boasting a top-30 singles player – but will have earned their 29th Davis Cup title if they account for Italy. Canada defeated Italy in the Davis
Cup round-robin matches in September, but bowed out to Finland in the knockout stage without Felix Auger-Aliassime and Denis Shapovalov, who have both battled
injury and form. The final is due to start at 2am AEDT on Monday and will be broadcast on 9GEM.\n\nSinner ends Djoker’s Davis Cup dream Novak Djokovic’s dream
of capping his great season by blasting Serbia past Australia in the Davis Cup final was blown up by Jannik Sinner’s extraordinary double triumph over the world
No.1.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nInstead, in the Davis Cup
final in Malaga, Australia will face an Italy team running hot on the heroics of their golden boy Sinner, who beat Djokovic in both the singles and the doubles
in the space of five exhilarating hours in their semi-final on Saturday (Sunday AEDT). Jannik Sinner saved three match points before going on to beat Novak
Djokovic. Credit: Getty Images In one of the great Davis Cup rubbers, Djokovic had held three match points against Sinner and was on the verge of a completing a
relatively workaday 2-0 victory for the Serbs.\n\nBut for only the fourth time in his unparalleled career, Djokovic was beaten from match-point up as Sinner
roared back to complete his second victory over the master in just 12 days, winning 6-2, 2-6, 7-5.', '‘Biggest win of my career’: De Minaur, Popyrin power
Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nThe 22-year-old Sinner, by now an irresistible force, then re-emerged to team up with
Lorenzo Sonego and spray massive forehand winners around liberally as they outplayed a by now weary-looking Djokovic and Miomir Kecmanovic 6-3, 6-4. Djokovic
had three match points in the singles rubber. Credit: Getty Images It was a crushing end to Djokovic’s hopes of ending his triumphant season, which has featured
three grand slam victories and an ATP finals showdown victory over Sinner just six days ago, with a second Davis Cup title. After telling fans to “shut up”
earlier in the week, he again got irritated with the crowd in the doubles, pretending to be the conductor to the Italian fans’ deafening chorus, but though he
may be the greatest player who ever picked up a racquet, doubles looks an alien game to him.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia
into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nHis serve was broken in the first set and his court positioning left a big hole for Sinner to blast
his forehand through the world No.1 and Kecmanovic, who had earlier worn down Lorenzo Musetti 6-7 (7-9), 6-2, 6-1 in the tie’s opening singles. The tie, though,
will forever be remembered for Sinner’s comeback triumph in their third brilliant singles duel in less than a fortnight. This time, the Sinner of the Turin
group stages re-emerged, the young titan reprising his victory of 12 days ago when he became the only man since Djokovic’s Wimbledon final loss to Carlos
Alcaraz to defeat the 24-time grand slam champion. Here he did something even more astonishing when, after soaking up immense pressure throughout the deciding
set, he stared at 0-40 on his serve at 4-5.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age,
2023-11-24T23:10:22+00:00)\nYet the 22-year-old proved absolutely nerveless, firing down an ace to save the third of the three match points, and then
immediately turning the tables by breaking Djokovic in the very next game amid wild Italian celebrations in the stands at the Palacio de deportes Martin
Carpena.\n\nSinner then served out for a famous win, ending Djokovic’s cherished unbeaten 21-singles winning streak in the Davis Cup and becoming the first man
since 2019 to beat the Serb in singles twice in the same year. The drama between the two rivals wasn’t over, as the two teams inevitably both drafted in their
top singles players into the doubles decider. Sinner was again magnificent and still looking ominously fresh after more than four hours on court, with
Djokovic’s weak backhand into the net off the Italian’s second serve in the doubles on match point seeming to sum up his disappointment. The beneficiaries of
all this?', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nThe beneficiaries of all
this? Lleyton Hewitt’s Australian side, who were happily enjoying a day’s rest at their team hotel further down the Costa del Sol, watching the drama unfold on
TV.\n\nWhile the Italy tie didn’t finish until 8.15pm after a noon start, Alex de Minaur and co should be fresh for combat in Sunday afternoon’s (Monday AEDT)
final as they seek to lift the Davis Cup for the first time in 20 years. AAP De Minaur, Popyrin power Australia into Davis Cup final Lleyton Hewitt has declared
his close-knit tennis band of brothers will give it “a hell of a shake” to bring back the Davis Cup to Australia after they roared into another final in
Malaga.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nFollowing their 2-0 victory
over Finland in Friday’s semi-final, executed superbly without a dropped set in singles wins for Alexei Popyrin and Alex de Minaur, team captain Hewitt said he
couldn’t be prouder of their achievement in making back-to-back finals.\n\nAnd even though they’ll be facing either a Serbian side led by world No.1 Novak
Djokovic or an Italian team inspired by young superstar Jannik Sinner in Sunday’s (Monday AEDT) final, the inspirational de Minaur was adamant: “No one really
wants to play us.” He delivered once more for Australia, taking an 11th victory in his last 13 cup singles matches, sealing Friday’s (Saturday AEDT) win by
defeating Emil Ruusuvuori 6-4, 6-3. Alex de Minaur stretches to return the ball against Finland’s Emil Ruusuvuori.', '‘Biggest win of my career’: De Minaur,
Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nCredit: AP It had followed a superb win from teammate Popyrin, who defied
jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even
needed.\n\nIt will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final,
seek a first win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves
to a deciding doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let
the doubles guys do the job.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age,
2023-11-24T23:10:22+00:00)\n“Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022. De Minaur had been the hero on
Wednesday, pulling out an extraordinary comeback win over Jiri Lehecka with Australia on the verge of defeat in their quarter-final with the Czech
Republic.\n\nBut he was happy to pass the laurels to Popyrin, a late squad replacement who’d been asked to take on the troublesome “second singles” role by
Hewitt. Loading “Alexei hasn’t played for Australia in a very long time [two years], had a massive match today against a very high-quality opponent and dug
super deep,” said de Minaur. Popyrin managed to deliver a rock solid game, making only 11 unforced errors to his erratic opponent’s 25, to prevail in just under
an hour-and-a-half. “It was probably the biggest win of my career,” he admitted.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis
Cup final (The Age, 2023-11-24T23:10:22+00:00)\n“It was probably the biggest win of my career,” he admitted. “To win a match that means so much to us, it’s an
honour and something I’ll never forget. Nothing compares to this.\n\n“I thought I was ready but when I came out and heard the two sets of fans chanting so hard,
but it was nerves that I’ve never experienced before.” Loading It wasn’t easy. The pressure was on with Thanasi Kokkinakis, the injured teammate he’d replaced,
Purcell and Jordan Thompson having all lost opening singles ties in this year’s competition, and Popyrin struggled too against a big-hitting 22-year-old going
for broke. But after repelling a set point in the opening stanza, Popyrin’s solidity in comparison to his increasingly erratic opponent helped silence a noisy
contingent of Finnish fans, who’d been hoping to cheer their underdogs to a first-ever final.', '‘Biggest win of my career’: De Minaur, Popyrin power Australia
into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)\nDe Minaur, cheered on by his excited mum, then didn’t need similar heroics to Wednesday’s as he
overcame the loss of the first two games.']
</item 1>

<item 2>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Australia, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Max Purcell, Matt Ebden, Popyrin, Davis Cup
final, De Minaur, Otto Virtanen
</item 2>

<item 3>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.

Graph facts:
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> AFLW
- Gold Coast Suns -LOCATED_IN-> Gold Coast
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> Gold Coast
- Comeback kings Richmond take the AFLW points in Cairns -MENTIONS-> AFLW
- Sydney Swans -PART_OF-> AFLW

Entities: Gold Coast, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Swans, North Sydney Oval,
Matildas, A-League Women, Blacktown, Swans’ first AFLW victory, A-League Women launch, Perth, Melbourne, Sydney, The Guardian
</item 3>

<item 4>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.

Graph facts:
- Livingstone -SAID-> Women’s World Cup
- Livingstone -PARTICIPATED_IN-> West Indies v England: first T20 international
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Livingstone
- West Indies v England: first T20 international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> first T20 international

Entities: Livingstone, Western Bulldogs, Molloy, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds,
Swans, AFLW competition, game time and scoring, AFL, The Guardian
</item 4>

<item 5>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.

Graph facts:
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Russia
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Russia
- Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. -MENTIONS-> Russia
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Russia
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Russia

Entities: Russia, The Age, Australia, Jannik Sinner, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Darren Cahill, Daniil
Medvedev, Lorenzo Sonego, Max Purcell, Matt Ebden, Italy, Davis Cup semi-final triumph over Serbia, 2022 Wimbledon championships, Popyrin, Davis Cup final
</item 5>

<item 6>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.

Graph facts:
- England choose survival over oblivion and a chance of summer trip to France -MENTIONS-> women’s football
- England thrash Scotland but cannot avoid Team GB Olympic heartbreak -MENTIONS-> women’s football
- Comeback kings Richmond take the AFLW points in Cairns -MENTIONS-> AFLW
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> AFLW
- Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds -MENTIONS-> Livingstone

Entities: women’s football, AFLW, Jack Snape, Harbour City, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Swans,
Chloe Molloy, Nicole Livingstone, North Sydney Oval, first AFLW match, winless campaign, Collingwood, AFL, Sydney, The Guardian
</item 6>

<item 7>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.

Graph facts:
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> Adelaide
- Comeback kings Richmond take the AFLW points in Cairns -MENTIONS-> Adelaide
- NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers -MENTIONS-> Adelaide
- Najwa Allen -WORKS_FOR-> Adelaide
- Chelsea Randall -WORKS_FOR-> Adelaide

Entities: Adelaide, Jack Snape, Sydney Swans, AFLW, Henson Park, Loretta Johns, Australian rules football, Harbour City, marquee pride match, Craft beer and
cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Collingwood, Sydney, The Guardian
</item 7>

<item 8>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, The Age, West Coast, AFLW, Gold Coast, Geelong, Comeback kings Richmond take the AFLW points in Cairns, Western Bulldogs, Melbourne,
Cairns, Richmond, North Melbourne, Swans, Chloe Molloy, AFL Photos
</item 8>

<item 9>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Following their 2-0 victory over Finland in Friday’s semi-final, executed superbly without a dropped set in singles wins for Alexei Popyrin and Alex de Minaur,
team captain Hewitt said he couldn’t be prouder of their achievement in making back-to-back finals.

And even though they’ll be facing either a Serbian side led by world No.1 Novak Djokovic or an Italian team inspired by young superstar Jannik Sinner in
Sunday’s (Monday AEDT) final, the inspirational de Minaur was adamant: “No one really wants to play us.” He delivered once more for Australia, taking an 11th
victory in his last 13 cup singles matches, sealing Friday’s (Saturday AEDT) win by defeating Emil Ruusuvuori 6-4, 6-3. Alex de Minaur stretches to return the
ball against Finland’s Emil Ruusuvuori.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Australia, Novak Djokovic, Jannik Sinner, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Alexei
Popyrin, Davis Cup final, Hewitt, Emil Ruusuvuori, Finland, Serbian side, Italian team, Davis Cup semi-final, Alex de Minaur
</item 9>

<item 10>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.

Graph facts:
- West Indies v England: first T20 international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> West Indies v England: first T20 international
- Livingstone -SAID-> game time and scoring
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> first T20 international

Entities: Livingstone, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Swans, Matildas, domestic
women’s sport, long-term growth, Perth, AFL, Sydney, The Guardian, Women’s World Cup
</item 10>
=== END ITEM id=be1fdd1b8508d78a ===
