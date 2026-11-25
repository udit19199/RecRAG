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

This file has 4 items.

=== ITEM id=34cac88e92a5c73e ===
Gold sentences and facts:
Sentence 0: We identify with his struggles and his little glimmers of connection and triumph — which makes him the perfect superhero for Thanksgiving.
  Fact 0.0: People identify with the superhero's struggles.
  Fact 0.1: People identify with the superhero's little glimmers of connection and triumph.
  Fact 0.2: Identifying with the superhero makes the superhero the perfect superhero for Thanksgiving.
Sentence 1: Apart from the food - arguably the main component of the day - the holiday is a beloved time of year when Americans come together to celebrate what they are thankful for, either with family or friends.
  Fact 1.0: Thanksgiving is a holiday.
  Fact 1.1: The food is arguably the main component of Thanksgiving day.
  Fact 1.2: Thanksgiving is a beloved time of year.
  Fact 1.3: Americans come together during Thanksgiving to celebrate what they are thankful for.
  Fact 1.4: Americans celebrate what they are thankful for with family or with friends during Thanksgiving.

Retrieved items:
<item 1>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
This season, NFL’s Thanksgiving Day games created history. Across all the three games, the NFL recorded an average viewership of 34.1 million. These figures
surpassed last year’s record viewership of 33.6 million. Furthermore, the combined unduplicated audience for all three games reached 133 million, marking the
second-highest total audience on Thanksgiving, surpassed only by last year’s numbers.

These figures paint a pretty one-sided picture, but the choice is much more nuanced. Comparing the NBA Christmas Day Games and NFL Thanksgiving Day Games comes
down to personal preference. Each tradition holds its own significance, adding to the diverse fabric of American sports culture. Whether one prefers the slam
dunks and three-pointers of the NBA or the hard-hitting action on the football field, both traditions offer a festive and entertaining way to celebrate holidays
with family and friends.

Are the NFL’s new decisions threatening the NBA’s Christmas Day dominance?
</item 1>

<item 2>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
However, you don’t need to have a big meal to celebrate your Friendsgiving, as there are many ways to change up your usual plans this year.

Here’s a list of activities that you could do for Friendsgiving that aren’t the typical sit-down dinner.

Shopping at local holiday markets

(Getty Images)

While Christmas is still over a month away, many cities in the US already have their holiday-themed markets up and running. And since Friendsgiving isn’t
necessarily held on the Thanksgiving holiday, why not use it as an opportunity to check out some new shops?

For example, in New York City, Bank of America is currently hosting its annual Winter Village at Bryant Park in Manhattan. At the beloved New York City spot,
there’s a wide variety of food vendors, so you and your friends can take your pick at what you want for dinner and dessert.
</item 2>

<item 3>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct. 20 on PlayStation 5. The game was reviewed using a pre-release download code provided by Sony Interactive
Entertainment. Vox Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via
affiliate links. You can find additional information about Polygon’s ethics policy here .
</item 3>

<item 4>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
A Haunting in Venice is his best yet, as Branagh’s confidence as director and performer in this mode only continues to grow. It’s perfect fall viewing.

Quiz Lady

Where to watch: Available to stream on Hulu

Genre: Comedy

Run time: 1h 39m

Director: Jessica Yu

Cast: Awkwafina, Sandra Oh, Will Ferrell

Sandra Oh and Awkwafina play a pair of estranged sisters who try to win big on a game show to pay off their mom’s debts. The supporting cast includes Will
Ferrell, Jason Schwartzman, Tony Hale, and the late Paul Reubens.

New on Prime Video

Freaks vs.
</item 4>

<item 5>
Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit (FOX News - Lifestyle, 2023-11-23T08:50:55+00:00)
But they have only a brief autumn harvest from late September to early November and a short shelf life.

Fresh cranberries were known only in a few areas of the nation that possess the right combination of sandy soil, long summers days and long winter dormancy.

SEN. MARSHA BLACKBURN ON THANKSGIVING BLESSINGS, INCLUDING FAMILY, FOOD, COUNTRY: ‘TIME TO EXPRESS GRATITUDE’

Urann turned this hyper-local fruit into an international industry — and changed the flavor of gratitude in the United States.

"Marcus Urann was very influential in expanding the cranberry market," Brian Wick, executive director of Massachusetts Cranberries, told Fox News Digital.

"And that influence is still felt today."

His influence is also felt today far from the soil and of the cranberry bog — in the halls of American academia.

‘Let the love of learning rule humanity’

Marcus Libby Urann was born on Oct.
</item 5>

<item 6>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
“Those slavering jaws; the lolling tongue; the rime of saliva on the grizzled chops.”

This is Angela Carter, not on Venom, but on the wolf from “Little Red Riding Hood” in The Bloody Chamber, her collection of reimagined folk and fairy tales.
It’s actually from “The Company of Wolves,” which is one of three retellings of “Little Red Riding Hood” in the collection, each with its own spin on the
familiar tale.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences. When we award the
Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth fitting into your
schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .
</item 6>

<item 7>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Sure, everyone in that room is trying to hold things together. Mary Jane wants to impress Norman, Harry wants to impress Norman and Mary Jane, Peter loves Mary
Jane but doesn’t dare hurt Harry, and Aunt May is finishing what’s presumably a dope turkey, in an attempt to care for all of them.

It’s the trying that counts: Life can be hard, weird, and cruel, but while sitting down for a meal with our nearest and dearest, maybe for at least a little
while, we won’t need to have it all figured out.

Not all Spider-Man Thanksgiving dinners end up falling apart, and some succeed in reminding Peter that he isn’t alone. The first season of The Spectacular
Spider-Man ends with a bang: One of Peter’s closest pals, Eddie Brock, has become Venom, and has threatened everyone Peter holds dear. This comes just after
Peter himself was infected by the symbiote, and went through the now-iconic throes of pushing everyone away.
</item 7>

<item 8>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Spider-Man is the only Marvel character to rate a balloon in the Macy’s Thanksgiving Parade. Other Marvel characters have appeared on floats, but only
Spider-Man has been inflated to 80 feet and pulled through the Upper West Side. (That also makes him the only Marvel character to have his head horrifically
torn open by a tree branch.)

From his first comic book appearance, Spider-Man has been a reminder that life is hard and complicated, and being a superhero doesn’t preclude anyone from
experiencing ordinary frustrations, setbacks, and confusion. But no matter how Spider-Man’s Thanksgiving escapades turn out, they remind readers and viewers
that the holiday is about the hope of mutual connection, support, and nurturing. Even if the girl of your dreams is out of reach and the Green Goblin is on your
case, a table, some friends, and Aunt May’s rad turkey might just make everything better for a little while.
</item 8>

<item 9>
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
</item 9>

<item 10>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Our best-of-the-year list reflects that range.

How the Polygon top 50 list works

Every year, the staff’s film fans create individual ballots to reflect their top movies, with an option to rank them numerically or just weight them. We use
these ballots to generate a collective list, weighted by strength of opinion to make sure a much-seen mainstream movie doesn’t have too much advantage over a
terrific but underseen indie. And then we put it all out there for your enjoyment. The top 10 picks on each our staff’s lists will be listed in the comments.

Any movie released in the US this calendar year is eligible, but since we are publishing this in early December, some December releases are underrepresented. We
hope you’ll find a new favorite here on our list of the best movies of 2023.

Honorable mentions Movies that received votes but did not crack our top 50: The Pope’s Exorcist, Bottoms, Rye Lane, No Hard Feelings, Hundreds of Beavers, Evil
Dead Rise, Guardians of the Galaxy Vol.
</item 10>

<item 11>
Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
Over the course of two-episodes, discover your new favorite comedian.

Available Nov. 29

American Symphony (Netflix Documentary)

Bad Surgeon: Love Under the Knife (Netflix Documentary)

From Netflix: This shocking docuseries about one of the biggest frauds in modern medical history exposes a thoracic surgeon famous for groundbreaking
achievements in the field of regenerative medicine.

Available Nov. 30

The Bad Guys: A Very Bad Holiday (Netflix Family)

From Netflix: Mr. Wolf and his crew of animal outlaws realize they’ll need to restore the city’s Christmas spirit to keep their annual holiday heist afloat.

Family Switch (Netflix Film)

From Netflix: Jess and Bill Walker are doing their best to keep their family connected as their children grow older, more independent, and more distant. When a
chance encounter with an astrological reader causes the family to wake up to a full body switch, on the morning of the most important day of each of their
lives, can the Walkers unite to land a promotion, college interview, record deal and soccer tryout?
</item 11>

<item 12>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
Going hiking

Depending on the weather outside, your plans for Friendsgiving could consist of going on a hike. Whether it’s at your local park or a trail that’s an hour or so
away from you, hiking is the perfect daytime activity for you and your friend group – no matter how big or small.

Decorating or baking sweets

(Getty Images)

Decorating or baking some sweet treats is another easy indoor activity. Whether you make a pie from scratch or simply frost your ready-bake sugar cookies, it’s
something interactive for you and your pals to do together.

You can also opt for treats that are on theme with the fall and winter seasons, from baking an apple crisp to decorating gingerbread cookies.

Running a Turkey trot

Whether it’s before, during, or after Thanksgiving Day, many cities in America host an annual Turkey Trot run.
</item 12>

<item 13>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 13>

<item 14>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
There’s so much going on in this patter-filled, joke-a-second comedy that it feels like 2023’s Everything Everywhere All at Once, right down to the big life
messages smuggled into all the goofery. It’s a high-speed joke-fest that doesn’t take Barbie any more seriously than she deserves — but does pay solemn homage
to all the ways, positive and negative, that Barbie fandom makes people feel. —TR

8. Spider-Man: Across the Spider-Verse

Directors: Joaquim Dos Santos, Kemp Powers, Justin K. Thompson

Cast: Shameik Moore, Hailee Steinfeld, Daniel Kaluuya

Where to watch: Netflix, or for digital rental/purchase

2018’s Spider-Man: Into the Spider-Verse changed the game for superhero movies and animated movies. It raised the bar for what a Spider-Man story could be, and
inspired animators to push the boundaries of what their movies could look like.
</item 14>

<item 15>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Every frame of the movie is a visual feast, just bursting with lovingly rendered details.

With a bigger cast and higher stakes, Across the Spider-Verse wasn’t going to finish its story in just one movie. It ends on a huge cliffhanger, but all the
pieces are in place for one hell of a trilogy finale. —PR

7. The Boy and the Heron

Director: Hayao Miyazaki

Cast: Soma Santoki, Masaki Suda, Ko Shibasaki

Where to watch: Theaters

Texture is essential to Hayao Miyazaki’s work. Sometimes the most visceral memories of his movies are the ways liquids glob or glide, while wood flecks flint
and flake in tiny explosions. Famously, even his still shots contain some motion, often featuring the slightest of movements that make the world feel more full
— more real.
</item 15>

<item 16>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Today’s 5.5% rate is definitely an improvement on last year’s 2.9%. But it’s nothing compared to the years preceding COVID-19, such as 2017, where we were
seeing growth of 18% or more.

Inflation is making an impact but not as much as the worry around consumer spending, said Adobe, which said that spending is being led actually by more
discounts to encourage buying, rather than less buying of more expensive products. That leads one to wonder what kind of impact that’s having on retailers’
margins.

“Cyber Week is off to a strong start with Thanksgiving driving a record $5.6 billion in online spend as consumers took advantage of strong discounts and
continued their shopping plans, virtually,” said Vivek Pandya, lead analyst, Adobe Digital Insights, in a statement. “Mobile shopping hit an all-time high, as
shoppers took to their smartphones to get the best deals during holiday gatherings, further solidifying mobile’s growing importance in e-commerce.”
</item 16>

<item 17>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
You play as both the yassified Peter Parker from Spider-Man Remastered and Miles Morales from his eponymous debut in 2020, switching between the two at your
leisure with all the speed afforded by the PlayStation 5’s SSD.

In fact, speed is one of the primary improvements on display in the first PS5-exclusive Spider-Man game. New to the series are web wings, which allow the
Spider-Mans to fly through the city, gliding over rooftops and gaining speed from wind tunnels without needing to rely on buildings, bridges, or other anchors
for web-slinging. (They’re especially useful when crossing the East River into the newly added Brooklyn and Queens.) Web-swinging is faster now as well,
enhancing that ineffable feeling of being Spider-Man as you glide through New York with ease. Whether you’re playing as Miles or Peter, traversing the three
available boroughs is an undeniable joy — so much so that I found myself using fast travel less frequently than in the previous two games, because movement felt
that much more fluid.
</item 17>

<item 18>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
For the occasion, everyone can bring a different type of wine – from red to white – and give a few sips from each bottle a shot. If you’re someone who’s a big
fan of wine, you could even end up discovering a brand that becomes your new favourite drink. When hosting the activity, you could make your own charcuterie
board – filled with different meats and cheeses – to pair with wine.

Playing a football game

If your friends have a fun, competitive side, you can take part in another outdoor activity. You and your friend group can host the event in any big outdoor
space you’d like, whether it’s a local park or your pal’s backyard.

In addition, football meets the typical theme of Thanksgiving, as a beloved tradition on the actual holiday is to watch a sports game with the family.

Hosting an early holiday gift exchange

Who’s to say that you can’t give your gifts to your friends a bit early this season?
</item 18>

<item 19>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
In a recent poll from the American Psychological Association, 41 per cent of US adults said their stress levels increased from November to January when compared
to other months of the year. Nearly 50 per cent of Americans also described their stress level during the season as “moderate”.

Everyone has different concerns leading up to Christmas, Hanukkah, or New Years - from family drama at the dinner table to a small spats with your mother about
decorations. With what can be stressful events, it only begs the question: do you really want to bring your partner into the mix? While it’s important to
eventually introduce your loved ones to your significant other, relationship experts believe there are some discussions you should have beforehand, especially
during the holidays.

Speaking to The Independent, Rachel Zar - a licensed marriage and family therapist at Spark Chicago Therapy - acknowledged that while every family is different,
there’s a variety of problems that could arise during the holiday season - mainly because we put too much pressure on having a joyful holiday.
</item 19>

<item 20>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Beleaguered movie theaters got a boost this year from film fans who made an event out of seeing two radically different movies back-to-back: a slyly satirical
movie about a living fashion toy, and a heavy, grim historical drama about the father of the atomic bomb. While Barbie and Oppenheimer do have a clear theme in
common, they’re still radically different movies. But the way they collectively dominated the 2023 box office and the cultural discussion is a strong reminder
that regardless of tone or topic, a great story, told passionately, stands out. Cinephiles can be drawn to just about any movie in any genre, if it’s well
crafted and engaging, no matter what kind of fruit it is.

Here at Polygon, we have pretty eclectic collective tastes: Some of us are hungriest for action, horror, epic fantasy, or challenging science fiction, while
some of us prefer complicated drama or dark neo-noir, and others are drawn to musicals, comedy, animated adventures, and other lighter fare.

Our best-of-the-year list reflects that range.
</item 20>

<item 21>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
While the winter holidays are still weeks away, you could use your Friendsgiving to host your annual gift exchanges.

For example, many friend groups in the US have opted for the budget-friendly activity of Secret Santa, where each person is randomly assigned someone to give a
gift to. During the exchange, you then reveal to your pal that you’re their “secret Santa,” before giving them their present.
</item 21>

<item 22>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Sound of Freedom

Where to watch: Available to rent on Amazon, Apple, and Vudu

Genre: Crime thriller

Run time: 2h 11m

Director: Alejandro Gómez Monteverde

Cast: Jim Caviezel, Mira Sorvino, Bill Camp

One of the most surprising (and controversial) box-office hits of the year, Sound of Freedom purports to be a true story about stopping child trafficking. The
truth is much more complicated than that.
</item 22>

<item 23>
Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
Even with Netflix’s recommendation algorithm serving you new movies, new TV shows, and original programming tailored to your viewing habits, the streaming
service’s firehose of content makes what’s coming and going difficult to parse.

It’s a great month for adult animation, with both the Scott Pilgrim animated show and Blue Eye Samurai hitting Netflix this month. Also, there’s the Squid Game
reality show — because right, that’s a thing — and a new season of The Crown. But this month, we’re highlighting The Social Network, with another new David
Fincher movie on its way to Netflix... but not for the reasons you probably think.

Editor’s Pick: The Social Network

Hey, did you know that there’s a popular fan theory that bestselling romantic comedy Red, White, and Royal Blue started off as fanfic of The Social Network?
Well, now you do!
</item 23>

<item 24>
The best surprise vampires in things you did not expect to see vampires (Polygon, 2023-11-15T18:02:00+00:00)
The clever folks behind Spider-Man got around the strict content standards of their network with a loophole that might be even creepier than how vampires
normally feed: by giving him little suckers on the palms of his hands that drain victims of “plasma.”

Hilariously, a creative application of vampirism would be how the show would later adapt another villain normally too violent for kids TV: Carnage, a serial
killing symbiote who now “drained” his victims’ “life force” instead of cutting them to bits.

Spider-Man: Across the Spider-Verse: Miguel O’Hara

Miguel O’Hara, the Spider-Man of 2099, is one of Spider-Man: Across the Spider-Verse’s big enigmas. Comics readers will know plenty about his whole deal, but
the movie doesn’t really explain much — except for a brief scene where he’s shown to have fangs, and an offhand reference that he is kind of a vampire? Sure,
he’s a jerk, but he’s still kind of cool.
</item 24>

<item 25>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
(It’s up.) Figuring out how to get there is another matter.

Even in games that prominently feature it, climbing is often functional, at best — whether it’s Link refusing to acknowledge handholds are a thing or Nathan
Drake snapping to ledges with a level of magnetism only thought possible at CERN.

Jusant is the first and only game I’ve played that gets it — that sees the inherent grace in the sport and portrays it on the screen, not as a means of getting
somewhere, but as the reason to go there in the first place. —Ari Notis

8. Marvel’s Spider-Man 2

Developer: Insomniac Games

Where to play: PlayStation 5

Marvel’s Spider-Man 2 builds on the successes of the first game, both narratively and in its immensely satisfying gameplay. The game’s story takes a familiar
Spider-Man narrative — that of friends turned foes — and adds new twists, building out compelling villains (and demanding ones) in the process.
</item 25>

<item 26>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

If there’s one thing we love here at Polygon, it’s the Halloween season.

We cover horror year-round, whether it’s the best horror movies you can stream at home or the best horror movies on Netflix, and generally, we have our finger
on the undead pulse of the newest and ghastliest releases in horror. We even have a list of the best horror movies of the year (ranked by scariness, of course).

Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.
</item 26>

<item 27>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
The spotted white beagle and his adolescent cohort – Charlie Brown, Marcie, Peppermint Patty, and many more – have always transferred easily from a panel on the
page to an animation frame, and this official adaptation has a trademark mix of wry humour, flights of fantasy, and heartwarming camaraderie. Happiness is
Holiday Traditions is a bonus special to close off the third season of the series, and it will stay true to the rule that helps make Peanuts endure: no adults
are seen or heard. Also on Apple TV+: The trailer for Family Plan (December 15), an action-comedy feature where Mark Wahlberg plays a former master assassin who
left the trade and secretly built a new life in the American suburbs as a cheerful husband and father, is goofy as all heck. The question is whether it’s
ludicrous fun or just sappy shenanigans?
</item 27>

<item 28>
Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit (FOX News - Lifestyle, 2023-11-23T08:50:55+00:00)
CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

American farmers grow twice as many cranberries as No. 2 Canada and about 20 times the production of third-place Chile, according to data from the Cranberry
Institute in Carver, Massachusetts, next to Plymouth.

The state of Wisconsin, meanwhile, surpassed Massachusetts decades ago as the nation’s top cranberry grower.

Six out of 10 Americans say that cranberry sauce has and always will be on their Thanksgiving table, while more than one third say they "can’t live without"
cranberry sauce on Thanksgiving, according to Ocean Spray.

Cranberry bogs remain a signature of the landscape in the land of the Pilgrims.

"I felt I could do something for New England. You know, everything in life is what you do for others." — Marcus Urann

"For Massachusetts and for this region, they’re so much a part of our history and our culture," said Wick.
</item 28>

<item 29>
Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit (FOX News - Lifestyle, 2023-11-23T08:50:55+00:00)
The tart, colorful cranberry's native habitat around the Pilgrim settlement in Plymouth, and its ascension to a uniquely vibrant place on America’s Thanksgiving
dinner table, are something of a coincidence — and a 20th-century phenomenon.

Credit cranberry king Marcus Urann, a true innovator of American agriculture.

"When the Pilgrims first set foot on Cape Cod … they may have well stepped on the American cranberry."

Urann, most notably, invented cranberry sauce — at least the commercial canned cranberry jelly central to Thanksgiving celebrations nationwide today.

The bookish boy from Maine became an influential college scholar and then a Boston attorney before giving up the bar for the bogs.

Urann purchased cranberry farms in Massachusetts early in the 20th century. He introduced a way to put cranberries in a can in 1912.

Cranberries are one of the few fruits native to the northern United States. But they have only a brief autumn harvest from late September to early November and
a short shelf life.
</item 29>

<item 30>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
The Holdovers is full of sudden twists, mostly backstory reveals suitable for a particularly startling stage play. But the real surprise is how personal and
specific it becomes, and excellent writing and acting help it dodge the expected parameters for this kind of story. Eventually, it settles into a three-hander
between Professor Hunham (Giamatti), his troubled adolescent student Angus (Dominic Sessa, in an intense star-making performance), and Mary (Da’Vine Joy
Randolph), the school’s head cook, an older Black woman mourning her son’s recent death in the military.

All three of these characters get room to develop and reveal themselves, and they’re all handled with sensitivity and real warmth that goes far beyond the usual
sentimental holiday feel-good fare. (If anything, this is a pretty dark story, and not in a laughable Bad Santa kind of way.) It’s a beautifully calibrated,
touching drama about starting over when there’s no other choice available. —TR

2.
</item 30>

<item 31>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Alex Hirsch’s paranormal comedy show about 12-year-old twins Dipper and Mabel, who solve supernatural mysteries in their great-uncle’s home in rural Oregon, is
the perfect entry point for young horror-loving audiences. It’s a pastiche of The X-Files, Twin Peaks, and Adventure Time all rolled into one, with the
freewheeling, funloving spirit of a summer vacation. There’s tons of fantastic episodes to choose from, but if you’re looking for an especially spooky one to
get in the Halloween spirit, you can’t go wrong with “Northwest Mansion Mystery.”

A ghost has taken up residence in the mansion estate of the wealthiest family in Gravity Falls on the eve of their annual party, and Dipper has been enlisted to
exorcize it. Upon arriving at the mansion, Dipper and popular girl Pacifica Northwest uncover a dark family secret that’s been buried for generations, one that
threatens to not only tarnish the family’s reputation, but engulf the entire town.
</item 31>

<item 32>
Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology (Scitechdaily | Science Space And Technology News 2017, 2023-11-21T20:46:17+00:00)
In recent decades, the cranberry industry has branched out into juices, snacks, and other products in pursuit of year-round markets. But for many people,
Thanksgiving is still the time when they’re most likely to see cranberries in some form on the menu.

Written by Serina DeSalvio, Ph.D. Candidate in Genetics and Genomics, Texas A&M University.

Adapted from an article originally published in The Conversation.
</item 32>

<item 33>
Watch great noir this Noirvember (Polygon, 2023-11-15T16:00:00+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

For some reason — blame the encroaching winter, post-Halloween boredom, or simple copycat syndrome, you be the judge — November has become a big month for
annual social-media-based collective challenges. From NaNoWriMo and NaPoWriMo to Movember and No-Shave November to, um, others, the extremely online crowd has
focused on November as a month for group events and group discussion.

For cinephiles, the fun one of the batch is Noirvember, a collective urge to watch and compare noir movies. Unlike most of these other challenges, Noirvember
doesn’t have an organizing website, a sponsor, a fundraising goal, or an established set of rules. (Though individual participants have certainly proposed
some.)
</item 33>

<item 34>
Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology (Scitechdaily | Science Space And Technology News 2017, 2023-11-21T20:46:17+00:00)
Analyzing the cranberry’s genome can indicate when it diverged evolutionarily from some of its relatives, such as the blueberry, lingonberry, and huckleberry.
Understanding how modern species evolved can teach plant scientists about how different traits are inherited, and how to effectively breed for them in the
future.

Cranberries and Thanksgiving

Cranberries’ close association with Thanksgiving was simply a practical matter at first. Fresh cranberries are ready to harvest from mid-September through
mid-November, so Thanksgiving falls within that perfect window for eating them.

Cranberry sauce was first loosely described in accounts from the American colonies in the 1600s and appeared in a cookbook for the first time in 1796. The
berries’ tart flavor, which comes from high levels of several types of acids, makes them more than twice as acidic as most other edible fruits, so they add a
welcome zing to a meal full of blander foods like turkey and potatoes.
</item 34>

<item 35>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
As for our heroes, Miles and Peter return from their individual outings for a dramatic team-up. Their two stories are threaded together, allowing for randomized
encounters where Peter might show up while you’re playing as Miles (or vice versa), with the two of you tag-teaming a random criminal and slamming them into the
ground in what feels like the world’s least fair fight. And yes, they do the Spider-Man-pointing-at-Spider-Man meme in-game. And yes, it is charming the first
time (though maybe less so the second). The main story sometimes swaps Spider-Mans mid-mission, as in the dramatic opening sequence where Sandman coats the city
in dust as Peter and Miles rush to help firefighters and first responders, with eerie echoes of 9/11. Side missions often allow either Spider-Man to tackle
them, but the main story beats require one or the other, as each hero is allowed his full particularity as a singular Spider-Man.
</item 35>

<item 36>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
The United States of America is home to several entertaining sports and leagues. However, football and basketball have resonated more deeply with the majority
of the population than other offerings. Since their inception, the two sports have captured the imaginations of the fans and captivated them with their
entertaining offerings—namely, the NFL and the NBA. The NBA and the NFL share a fierce rivalry because they stand out as the two preeminent professional sports
leagues in the United States. While the NFL enjoys greater popularity within the United States, the NBA far outperforms its counterpart in global popularity and
impact.

America’s Favorite Video Today

Both leagues have crafted their own festive traditions with the NFL’s Thanksgiving Day games and the NBA’s Christmas Day matchups. These special events have
become cherished customs, captivating fans in their unique ways. Thanksgiving Day games are as intertwined with Thanksgiving as turkeys, while Christmas Day
games have become as synonymous with Christmas as eggnog and presents.
</item 36>

<item 37>
The best surprise vampires in things you did not expect to see vampires (Polygon, 2023-11-15T18:02:00+00:00)
Preacher: Cassidy

Preacher, the comic book by Garth Ennis and Steve Dillon — and its subsequent TV adaptation from Sam Catlin, Evan Goldberg, and Seth Rogen — starts with the
basic assumption that God is real, and has walked out on the job. A lot of things are fair game at that point, namely angels, demons, and the like, but it’s
still a shock when you meet Cassidy. A brawling, hard-drinking Irishman that’s already a load of fun when you meet him, Cassidy shoots up the power rankings
when he sinks his teeth into a poor sap, because he’s a vampire.

Spider-Man: The Animated Series: Morbius

Hands down the ballsiest Spidey villain to adapt in a cartoon for children that censors wouldn’t even let use the words “sinister” or “kill,” Morbius — the
Living Vampire — frankly had no business being in a Saturday morning lineup.
</item 37>

<item 38>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
In addition, vendors are also selling a range of goodies – from ornaments to jewellery to scarves – which is the perfect chance to do some early holiday
shopping.

Hosting a movie marathon

Instead of hosting a big dinner, you could opt for another budget-friendly activity at your home: hosting a movie marathon. With the help of your friends, you
can make a list of any horror, romance, or comedy movies that you want to watch for the day. You could also top off the activity off by supplying an assortment
of snacks for your pals, or by cooking a meal all together.

Your movie options can also match the theme of Thanksgiving, with films like A Beautiful Day in the Neighborhood (2019), Toy Story (1995), Addams Family Values
(1993) and A Charlie Brown Thanksgiving (1973). If you want, you could add some Christmas movies to the mix, including Elf (2003), It’s A Wonderful Life (1947)
and Steel Magnolias (1989).
</item 38>

<item 39>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
New on Prime Video

Freaks vs. the Reich

Where to watch: Available to stream on Prime Video

Genre: Superhero/circus war movie

Run time: 2h 21m

Director: Gabriele Mainetti

Cast: Claudio Santamaria, Aurora Giovinazzo, Pietro Castellitto

This offbeat Italian superhero movie follows a group of circus performers in World War II who are sought after by the Nazis and team up to stop them. I have
heard it’s funny, sweet, and has strong action — definitely on my weekend watchlist.
</item 39>

<item 40>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
inflated Getty Macy's Thanksgiving Day Parade takes place in New York Sinclair's DINO balloon flies as the parade passes down 6th Avenue Reuters Macy's
Thanksgiving Day Parade takes place in New York An NYPD officer stands by AP Macy's Thanksgiving Day Parade takes place in New York The Charlie Brown balloon
flies in the parade Reuters Macy's Thanksgiving Day Parade takes place in New York A marching band takes part in the parade Reuters Macy's Thanksgiving Day
Parade takes place in New York A balloon flies in the parade Reuters Macy's Thanksgiving Day Parade takes place in New York A performer in the parade AP Macy's
Thanksgiving Day Parade takes place in New York The Ronald McDonald balloon flies in the parade AP Macy's Thanksgiving Day Parade takes place in New York A
clown performs in the parade Reuters Macy's Thanksgiving Day Parade takes place in New York The Olaf balloon is inflated Getty Macy's Thanksgiving Day Parade
takes place in New York Flag wavers take part in the parade Reuters Macy's Thanksgiving Day Parade takes place in New York Clowns on roller blades perform in
the parade Reuters Macy's Thanksgiving Day Parade takes place in New York The Charlie Brown balloon is
</item 40>

<item 41>
Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit (FOX News - Lifestyle, 2023-11-23T08:50:55+00:00)
Urann himself became an attorney in Boston after graduation. But in 1906, he decided to give it up to buy cranberry bogs in and around Cape Cod and Plymouth.

'Simple, insignificant-looking plant’

Pilgrim Edward Winslow provided our only contemporary account of the first Thanksgiving. He mentions only fowl and deer, and "our harvest being gotten in."

The harvest certainly included corn, scholars believe.

The rest of the "harvest" upon which they feasted is unknown.

"But there is no evidence the Pilgrims had cranberries at the first Thanksgiving," said Wick of Massachusetts Cranberries, despite the fact they are one of the
few fruits native to North America that grow in the northern United States.

It might have just been overlooked — as authors Cox and Walker suspect.

"There is no evidence the Pilgrims had cranberries at the first Thanksgiving."
</item 41>

<item 42>
Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit (FOX News - Lifestyle, 2023-11-23T08:50:55+00:00)
"There is no evidence the Pilgrims had cranberries at the first Thanksgiving." — Brian Wick

They cite colonial settler John Josselyn, who wrote in 1672 that the cranberry is nothing but "a small trailing plant" and a humble vine "that grows over Salt
Marshes that are overgrown with moss."

Benjamin Eastwood, touted by Cox and Walker as one of the cranberry’s "greatest promoters," spoke only in humble terms of his beloved fruit.

He called the cranberry a "simple, insignificant-looking plant" and a "stunted barren thing."

Urann saw the cranberry differently. He saw a global industry crawling along the bogs of Massachusetts.

"After he set up cooking facilities at a packinghouse in Hanson, Massachusetts, he began to consider ways to extend the short selling season of the berries,"
Smithsonian Magazine reported in a 2013 history of cranberries.
</item 42>

<item 43>
Super Mario Bros. Wonder is a perfect end for the Switch (Polygon, 2023-10-18T13:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

But, if you’re curious about the “Wonder” — if you’re not too precious about video game tradition — you can find within each Wonder stage a Wonder Seed, a
Mario-sized ball of botany that morphs the environment into something indeed quite wonderful. Like a musical number.

To choose wonder is to flood the atmosphere with lava-lamp plasma, send gravity into convulsions, or transform your hero into any variety of creatures — or even
inanimate objects. As my stoned college roommate once postulated while ripping an apple bong: “You always see the scenery, but have you ever wanted to be the
scenery?” Reader, he dreamt so we could live. For a moment, Mario gets to be the background.

The brilliance of Super Mario Bros.
</item 43>

<item 44>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
We’re coming up for air from our coverage of The Game Awards to loop our readers in on some of the best deals happening this weekend. If you’re looking to
experience some of the winners from the 2023 Game Awards for yourself, you can find discounts on a variety of titles on Steam. Some other highlights you might
want to check out include a sitewide sale on Blu-Rays from The Criterion Collection, and the Xbox Series X matching its Black Friday price on Walmart and
Amazon.

As we do every weekend here at Polygon, we’ve rounded up a collection of our favorite deals from the worlds of gaming and entertainment, along with some of the
best-selling products that have made a recent appearance on our site.

The best gaming deals this week

Steam is discounting many contenders from the 2023 Game Awards and winners from the past nine years.
</item 44>

<item 45>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
The Kill Room

Where to watch: Available to rent on Amazon, Apple, and Vudu

Genre: Dark comedy thriller

Run time: 1h 38m

Director: Nicol Paone

Cast: Joe Manganiello, Samuel L. Jackson, Uma Thurman

Joe Manganiello stars as a hitman turned artist in this funny comedy about how the worlds of fine art and high crime aren’t so separated after all. When he
turns to art as a method of laundering money, the hitman becomes an unexpected overnight sensation in the high-art scene.
</item 45>

<item 46>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
But how?

Compared to the lack of interactivity in 2018’s Spider-Man, MJ is now capable of tackling the enemies around her. In the first title, all she could do was to
move from one point to the other. Now, she’s able to do more than that – make pretty good and lethal moves. To be precise, the devs have given MJ a little
firepower, if not in too much capacity. Now, she’s got weapons instead of being clueless in moments of being vulnerable against armed enemies. Yes, she hasn’t
got any superpowers, but she’s got other means to help herself out.

Due to such involvement, players have now got a reason to be more invested into a given mission focusing on MJ. It wasn’t the case at all taking into
consideration her previous iteration in Insomniac’s original game. Mary Jane can now handle the circumstances in her own manner, thus breaking free from the
linear approach of just trying to hide and escape.
</item 46>

<item 47>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
With the recent release of Marvel’s Spider-Man 2, Insomniac has witnessed a huge appreciation from fans for delivering a memorable sequel. Not to mention the
epic portrayal of Peter Parker and Miles Morales fighting against some of the strongest villains. But then, there’s MJ as well who has made a comeback from the
first installment. And it’s her arc that has actually received an impressive elevation this time. Especially when considering the poorly received missions
featuring Mary Jane in the original game from 2018.

America’s Favorite Video Today

So what’s actually changed in 2023 that has created a vast difference? There are certain key reasons making Mary Jane in Spider-Man 2 a treat to watch. Not just
from the perspective of her relationship with Peter, but her own missions showcasing a different side this time. One can even call her the third unofficial
superhero in the latest PS5 exclusive.
</item 47>

<item 48>
This horror flick boasts the best opening 13 minutes of a film you’ll see this year (The Age, 2023-11-15T18:30:00+00:00)
Despite Thanksgiving Day’s heavy baggage (like Australia Day, many in the US have increasingly questioned the holiday’s ties to colonial aggression), the film’s
key creative breakthrough for Roth and Rendell came from something decidedly more contemporary: watching online “trample videos” from Black Friday shopping
sales, traditionally held the day after Thanksgiving, that started proliferating on social media around a decade ago with news reports and clips of shoppers in
a consumerist frenzy. “These Black Friday sales, they were never a thing when I was a kid. But now I know they’re even in Australia, they’re everywhere,” says
Roth, who saw fertile horror territory in the holiday’s materialistic shift. “The whole idea of Thanksgiving is you’re supposed to be thankful for what you have
and for your health, but then you just go out and kill your neighbour for a waffle iron.”
</item 48>

<item 49>
Dropout’s dropping CollegeHumor on its 5th anniversary — here’s why and what the future holds (Polygon, 2023-09-26T17:35:00+00:00)
And here we are. It’s kind of serendipitous and magical.

Mulligan: Yeah. It’s very special. The impostor syndrome is very real, but I find myself bouncing back and forth between the feeling of — When Wayne Brady says
something like that, you sort of can’t accept something that nice. So you deflect it away. But also, I feel like there’s something that would be too glib and
dismissive to not be cognizant about the incredible fortune that we’ve had.
</item 49>

<item 50>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
But fans see a lot of nuance within that formula, as Anderson’s voice (especially his sense of humor) develops from film to film.

In Asteroid City, his ridiculously meta story-within-a-story sci-fi film about an alien encounter, that voice hones in on the question of art and creativity —
who it’s for, what it brings the artist and the audience, why any form of recognition or acclaim is good enough for one creator while another strains to find
connection and resonance in their work. It’s the kind of film that moves so quickly, and with so little attempt to hold the audience’s hands and tell them what
to feel, that it takes some work to scratch the surface.

But it’s worth diving into the movie’s connections and themes, as a who’s who of actors — many from Anderson’s usual stable, and some debuting here — bounce off
each other, looking for meaning in an isolated desert setting.
</item 50>

<item 51>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
ADVERTISEMENT Article continues below this ad

The Milwaukee Bucks and the New York Knicks get the festive proceedings underway at Madison Square Garden. Furthermore, the Warriors are scheduled to face the
reigning champions, the Denver Nuggets. Apart from these two intriguing matchups, the league has also scheduled Lakers vs Celtics, Heat vs Sixers, and Suns vs
Mavs for Christmas. With only a few hours left until the tip-off, fans can kick back and enjoy the NBA with friends and family on the joyous occasion that is
Christmas.

Watch this Story: Chris Paul annihilates Scott Foster after the controversial call, ‘Don’t use a tech to get your point across’
</item 51>

<item 52>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Aunt May has just gotten out of the hospital, and even noted jerk “Flash” Thompson has given Peter hell for how awful he’s been acting.

Peter, attempting to make things right, opts to cook the Thanksgiving meal all by himself, but mostly succeeds in ruining the kitchen. No worries — Gwen Stacy
and her father, along with a recuperating Aunt May and her doctor, all help out, and the episode concludes with a pleasant dinner. There are no big turns or
twists, aside from Aunt May revealing that she’s publishing a cookbook. It’s just simple, earned solace in a life marked by chaos. Peter even gets a kiss from
Gwen on his porch, a sequence conducted with a John Hughes sense of satisfying romantic flair.

Various Spider-Man comics have also dabbled in seeing what Thanksgiving looks like when you’re young and arachnid-themed, but real life has associated
Spider-Man with the holiday, too. Spider-Man is the only Marvel character to rate a balloon in the Macy’s Thanksgiving Parade.
</item 52>

<item 53>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
But feeding friends and family is perhaps one of the most humane acts we can pursue. It’s less about the spirit of giving, and more about admitting that people
have innate, basic needs, like food and social comfort, and that those needs are best fulfilled when people work in tandem. Honestly, we should try to do it
more on every other day of the year: The number of people volunteering to help feed the homeless and families in need peaks around November and December, but
that energy needs to be carried through the preceding 10 months, too.

Spider-Man represents these needs, even though very few of us have to balance photojournalist work, a fraught dating life, and pummeling Dr. Octopus. Peter
Parker is often the most financially strapped among his Avengers buddies, and frequently the loneliest, too. Those are recognizable traits among many young
people, even those without radioactive-spider powers. When you’re growing up and trying to figure out the world, it’s easy to feel lost and isolated.
</item 53>

<item 54>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Outlaw Johnny Black

Where to watch: Available to rent on Amazon, Apple, and Vudu

Genre: Satirical Western

Run time: 2h 10m

Director: Michael Jai White

Cast: Michael Jai White, Anika Noni Rose, Erica Ash

Michael Jai White’s long-awaited follow-up to Black Dynamite is finally here: a “West-ploitation” movie about an outlaw who pretends to be a preacher and
settles in a new troubled town. The star and director spoke to us at length about the movie and the long road it took to get here.
</item 54>

<item 55>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
perform in the parade Reuters Macy's Thanksgiving Day Parade takes place in New York The Charlie Brown balloon is inflated Getty Macy's Thanksgiving Day Parade
takes place in New York The Tom Turkey float is in place to start the parade Reuters

Later, US Presidents including George Washington, John Adams and James Madison called for days of thanks throughout their presidencies.

However, it was not until 1863, during the Civil War, that President Abraham Lincoln declared a national Thanksgiving Day would be held each November to be
celebrated by all of the country.

When is Thanksgiving?

Thanksgiving takes place each year on the fourth Thursday of November.

This year, Thanksgiving falls on 23 November.

Why do Americans celebrate it and what do they do?

Although Thanksgiving may originally have had religious significance, the day has become a mostly secular holiday.

Most Americans consider the holiday a day to gather and express their thanks through food, family, and football – with multiple NFL teams playing on the
holiday.

During some Thanksgiving celebrations, people write down what they are thankful for and then read aloud from the pieces of paper.
</item 55>

<item 56>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The game is utterly devoted to giving the players the feeling of playing a hero in the Marvel universe with tons of cool powers and power categories based on
key Marvel heroes like Spider-Man and the Hulk. The system is easy, streamlined, and really carries the four-color comic book flavor. Probably one of the
coolest aspects of the game is that while there is an exhaustive list of Marvel heroes and villains at the back of the book, you can create your own hero with
their own power set and use that “bestiary” of heroic and villainous figures as allies and enemies in your own heroic adventures. For those of us who’ve always
wanted to make their own X-Gene Mutant, Spirit of Vengeance, or Spidersona, this is a must-have book that will bring hours of enjoyment.

Masks of the Masks

Superhero comics, for all their many flaws, remain my life’s great passion.
</item 56>

<item 57>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
One of the Miles-specific side quests involves learning about the artistic history of Harlem, and it entwines the real-life histories of Josephine Baker,
Langston Hughes, and Charlie Parker with the fictional history of SHIELD and other Marvel properties. An early mission shows off Insomniac’s technical chops
with a section that cleverly borrows a trick from Ratchet & Clank: Rift Apart. Of the two Spider-Mans, it’s Miles who feels more alive with the possibility to
break free of expectation and canon.

[Ed. note: Spoilers follow for Spider-Man: Across the Spider-Verse.]

The recent Spider-Man: Across the Spider-Verse has an obsession with “canon events,” pivotal moments in a Spider-Man’s life that the film’s villain, Spider-Man
2099, is dedicated to maintaining across the multiverse.
</item 57>

<item 58>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Here’s everything new to watch this weekend!

New on Netflix

Spider-Man: Across the Spider-Verse

Where to watch: Available to stream on Netflix

Genre: Superhero action

Run time: 2h 20m

Director: Joaquim Dos Santos, Kemp Powers, Justin K. Thompson

Cast: Shameik Moore, Hailee Steinfeld, Oscar Isaac

The highly anticipated follow-up to 2018’s Spider-Man: Into the Spider-Verse sees Miles Morales facing off not only against a dimension-hopping nemesis in the
form of the Spot, but a whole multiverse of Spider-Mans, Spider-People, and even a Spider-Dinosaur as he attempts to save the day once again.

From our multiversal review:

Not every theme and plot and moment in Across the Spider-Verse lands, particularly with the other part of this story still most of a year away.
</item 58>

<item 59>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Whether it’s the upside-down kiss, the death of Uncle Ben or Jefferson Davis, or simply the spider bite that starts it all, Spider-Man 2099 believes that
upholding these canon events, even the painful ones, is key to preserving the stability of each strand of reality. Miles disagrees, believing it is Spider-Man’s
job to resist the seemingly inevitable — to always do what’s right, not simply what’s preordained. The film ends with a promise of pushing back against
Spider-Man 2099, and with it, the established canon.

I was hungry for Spider-Man 2 to push back in a similar way — to treat the established story like Angela Carter repurposing “Little Red Riding Hood” to her
whims, like Michael Cunningham plumbing “Rumpelstiltskin” for a story about the desire to have a child, or like Helen Oyeyemi and her body of work.
</item 59>

<item 60>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Wingwomen

Where to watch: Available to stream on Netflix

Genre: Action

Run time: 1h 56m

Director: Mélanie Laurent

Cast: Mélanie Laurent, Adèle Exarchopoulos, Isabelle Adjani

Mélanie Laurent (Inglourious Basterds) is both behind and in front of the camera in this action comedy about women thieves on the run looking to pull off one
last job.

New on Hulu

A Haunting in Venice

Where to watch: Available to stream on Hulu

Genre: Horror mystery

Run time: 1h 43m

Director: Kenneth Branagh

Cast: Kenneth Branagh, Michelle Yeoh, Jamie Dornan

Kenneth Branagh’s Poirot adaptations have generally been a fun time, even when they have problems (looking at you, Death on the Nile).
</item 60>

<item 61>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“Sometimes it can be helpful for them to know about the dynamics that they’re stepping into and how they can best support you. Frankly, they should be aware if
there is tension or if there is stress within your family.”

(Getty Images)

Communication before the holidays goes both ways, and there are things that your family should know about your significant other too. “If there’s something
about your partner that could be surprising to your family, or that’s a notable difference from who someone you’ve brought home before, tell your relatives
that,” Zar said. “Let’s set your family and partner up for success.”

Before bringing your significant other home for the holidays, you may want to express to your family how you wish for them to feel included - such as
incorporating your partner’s family traditions into your own celebration. However, Zar advised against oversharing information before Christmas. You still want
your family and partner to connect in person, rather than hearing about each other through the grapevine.
</item 61>

<item 62>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Americans started preparing their Thanksgiving menus weeks ago, with the requisite turkeys and stuffing ingredients likely already purchased for the holiday,
which is celebrated this year on Thursday 23 November.

Apart from the food - arguably the main component of the day - the holiday is a beloved time of year when Americans come together to celebrate what they are
thankful for, either with family or friends.

In America, Thanksgiving is a cultural holiday that symbolises peace, thankfulness, and the beginning of the holiday season.

What is Thanksgiving and where did it come from?
</item 62>

<item 63>
Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit (FOX News - Lifestyle, 2023-11-23T08:50:55+00:00)
Plucky little cranberries have grown for millennia amid sandy soils and short summers of southeastern Massachusetts and Cape Cod.

The Pilgrims explored and then settled this same area in 1620 — and celebrated the first Thanksgiving the following year.

"When the Pilgrims first set foot on Cape Cod, even before they saw Plymouth Rock, they may have well stepped on the American cranberry," Richard S. Cox and
Jacob Walker write in their book, "Massachusetts Cranberry Culture: A History from Bog to Table."

Their 2012 book is a romantic ode to the quirks of New England history, geography and culture — and of its low-growing fruit.

MEET THE AMERICAN WHO GAVE THE NATION OUR THANKSGIVING ORIGIN STORY: PILGRIM EDWARD WINSLOW

It "seems doubtful," however, that the Pilgrims even noticed the cranberries.

"For all its tasty charms, the cranberry is hardly the sort to attract attention," Cox and Walker say in their book.
</item 63>

<item 64>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
Where is it celebrated?

Thanksgiving is one of the the most important cultural holidays in America, however, Canada also has its own Thanksgiving Day, on the second Monday in October,
and Liberia celebrates Thanksgiving on the first Thursday of November.

What do people eat on Thanksgiving?

In America, turkeys are an integral part of Thanksgiving Day, with most dinners including the bird.

An estimated 46m turkeys are killed annually for the holiday, however, two turkeys are pardoned each year by the president.

But Americans who don’t eat meat or follow a vegan lifestyle don’t have to worry about missing out, as there are numerous vegan and vegetarian Thanksgiving
options.

Americans also indulge in Thanksgiving favourites such as yams topped with marshmallows, stuffing, cranberry sauce, cornbread, and pumpkin pie.
</item 64>

<item 65>
At Christmas, Don McLean sings the praises of 'Silent Night,' other holiday classics: 'My mother's favorite' (FOX News - Lifestyle, 2023-12-17T09:00:48+00:00)
McLean said, "I want to share something on this point. Jack Nicholson, who is a very smart man, said something very important here. He said, 'I don't think the
American public wants to be moved anymore.' He doesn't think people want to feel things. And I am here to tell you: They do feel things, but they don't have
anybody out there who's giving them anything to feel."

He went on, "It's all spectacle and empty. For example, we have a terribly, terribly empty political situation right now. I've never seen anything like it in my
life. Everywhere you look, there's an empty suit. Nobody's got any brains. They don't know how to use the English language — it's pathetic."

McLean said that "this is where music comes in. It can help you. It can help you feel better — but we have to get over things.
</item 65>

<item 66>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
One can even call her the third unofficial superhero in the latest PS5 exclusive.

A refreshing take on Mary Jane’s stealth missions makes her a strong character in Marvel’s Spider-Man 2

ADVERTISEMENT Article continues below this ad

In the first title, stepping into MJ’s shoes was a great concept, but the devs somehow couldn’t execute the plan flawlessly. As a result, missions featuring her
as a playable character were heavily criticized by most of the fans. There was nothing too special about those quests whatsoever.

Now, looking at Mary Jane in Marvel’s Spider-Man 2, there’s a very clear difference one can make out. In short, it feels comparing night and day. Being a
journalist at The Daily Bugle, her character is bound to go places in order to investigate mysteries. The first game had such situations, but the sequel
improved on them. It gives MJ a lot more strength instead of being someone who always needs to be rescued in the last moment. But how?
</item 66>

<item 67>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 67>

<item 68>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Coming back to the ‘Wake Up’ mission, she’s seen sneaking past the hunters but she’s also capable of performing Takedowns on them. It wasn’t the case in the
previous title. Her missions have become better now with the devs actually spending time on giving her a chance to fight.

Coming back to the stun gun part, MJ is pretty skilled in using it, during missions like ‘Good Men’, thanks to her training with Sable. Also, now she’s got the
option to even run away and hide when spotted by enemies. The playability factor around her has increased a lot compared to being just an observer, for the most
part, in the first game. That’s the biggest and most visible difference.

Mary Jane and her missions in Spider-Man 2 allow the story to proceed without causing a problem for the central protagonists

The best thing about going through the quests showcasing Mary Jane is that they never try to overpower the ongoing story itself.
</item 68>

<item 69>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
However, when comparing the two festive traditions, which one holds more significance, or which tradition has garnered greater popularity?

How do NBA’s Christmas Day games compare to NFL’s Thanksgiving Day games?

ADVERTISEMENT Article continues below this ad

Both traditions hold a special place in the hearts of fans, creating an annual spectacle that goes beyond the typical sporting event. The NBA’s Christmas Day
games is a tradition which is even older than the NFL’s Super Bowl. The first Christmas Day game dates back to 1947, when the league featured three games as a
festive treat for basketball fans. Since then, the NBA Christmas Day has only grown in stature and significance. Last season, Christmas Day games averaged 4.27
viewers across all platforms. The significance of this tradition can be evaluated by the fact that these games attract about 200% more viewers than any other
regular season game.

USA Today via Reuters Credit: Kevin Jairaj-USA TODAY Sports

However, despite these impressive viewership numbers, the NFL’s Thanksgiving Day games eclipse them by a considerable margin.
</item 69>

<item 70>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Marvel built its comic book revolution on the back of one idea: rendering colorful superheroes as relatable. In the mid 20th century, DC do-gooders were
essentially square-jawed Sunday school teachers, which let Marvel corner the market on heroes with human depth and fragility. The Fantastic Four were vulnerable
to insecurity and in-fighting. The X-Men represented the cost of bigotry on a wide scale.

But Spider-Man represented this new wave best. Consumed by youthful everyman angst and desperate to find balance in his life, Spider-Man is broadly sympathetic.
We identify with his struggles and his little glimmers of connection and triumph — which makes him the perfect superhero for Thanksgiving. And Spider-Man
writers know it.

Thanksgiving has a complicated history, tied to the roots of American colonialism, and more recently processed through a lens of joyful capitalism. But feeding
friends and family is perhaps one of the most humane acts we can pursue.
</item 70>

<item 71>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
While it’s often viewed as a day for Americans to express what they’re grateful for, that doesn’t mean that everyone will celebrate. More specifically, the
event has been viewed as deeply controversial over the years, due to the contentious history of the first Thanksgiving celebration in the 17th century.

The holiday has been linked to instances of cultural appropriation, with some Americans believing that its history is frequently white-washed, prompting them to
condemn or ignore Thanksgiving as a whole.

However, the holiday itself has inspired another tradition of Friendsgiving – which can be held before, during, or even after the 23rd of November this year.
The activity typically consists of a celebration with close pals since some Americans don’t have the opportunity to return home for the holidays.

Over the years, people have celebrated Friendsgiving in a similar vein as Thanksgiving, with everyone sitting around the table for one big feast. It’s also
often done in a potluck style, with each guest bringing their own dish to contribute.
</item 71>

<item 72>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Maybe Hollywood as a whole took away the wrong lesson (we’ve seen so many multiverse movies), but Into the Spider-Verse redefined genre staples.

The sequel continues to raise the bar. It’s a little busier than the first, pulling in more characters, more frantic visuals, more emotional throughlines, but
it’s still bold, daring, and unlike anything else in the superhero movie sphere.

As the characters bounce across the multiverse, the animation style shifts, with each character rooted in a different style. Spider-Gwen’s world is rendered
with watercolors, which shift according to the tone of her story. She gets a step up in this movie — a second chance for Gwen Stacy, whose previous iterations
across media have not ended well. Meanwhile, Spider-Punk, the brash counterculture rebel, is made up of collaged bits, and animated against the norm so he
stands out even more. Every frame of the movie is a visual feast, just bursting with lovingly rendered details.
</item 72>

<item 73>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Peter’s plot involves a retelling of the Venom origin story, with beats that will be recognizable to those even passingly familiar with the Spider-Man mythos.
If Spider-Man 2 is a game of retellings, Peter’s story clings most closely to the canon. If you’ve seen Sam Raimi’s Spider-Man 3, you’ll know the gist of what
to expect when the symbiote finds its way to Peter. It isn’t until the late game that Peter’s story starts pushing more strongly against our canonical memories
and into something more unique to Insomniac.

Miles’ story takes more liberties. Hailey Cooper, Miles’ love interest from his solo game, returns here, taking more of an MJ-esque role and carving out her own
identity separate from Miles’. American Sign Language is featured heavily throughout Miles’ missions because of Hailey’s presence, feeling as natural and
integral to the story as anything from the canon.
</item 73>

<item 74>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
If you want curated lists of our favorite media, check out What to Play and What to Watch .

Retellings are compelling because they allow writers to toy with readers’ preconceived notions of how the story “should go,” both in terms of genre conventions
and in terms of plot. We know Little Red Riding Hood will go to her grandmother’s house, where she will find that her grandmother has been eaten by a wolf, just
as we know Spider-Man will be bitten by a radioactive spider. The fun, both for the author and the reader, comes in deviating from those norms — in finding new
ways to tell old tales.

With Marvel’s Spider-Man 2, Insomniac Games attempts something similar.

Mainstream Spider-Man stories are increasingly Spider-Men stories, and Spider-Man 2, the third Spidey game from Insomniac, is no different.
</item 74>

<item 75>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
The Macy's Thanksgiving Day Parade takes place in New York City

The day is also celebrated with the annual Macy’s Thanksgiving Day Parade in New York City.

The world’s largest Thanksgiving parade includes giant balloons of cartoons that float above the city sky, as well as marching bands and dancers.

Following Thanksgiving, the month-long shopping for the winter holidays begins, with Black Friday kicking off the season.

Why is Thanksgiving controversial?

Although many Americans view Thanksgiving as a beloved holiday, it is viewed by many who have abandoned the holiday as a erroneous celebration of what was
actually the conquest of Native Americans and Native American lands. The celebration has become controversial, as Thanksgiving has a contentious history that
goes far beyond when the first feast was held.

In addition to a holiday steeped with cultural appropriation, the period of history in America is frequently white-washed, which leads some Americans to ignore
the holiday, or, to instead observe a National day of Mourning.

Where is it celebrated?
</item 75>

<item 76>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
But in the end, the theme of the Spider-Verse movies is shaping up to be a story about people trying to be bigger and bolder themselves, trying to reach beyond
what they’re told they’re capable of, and do more. It’s no wonder that every part of Across the Spider-Verse is an attempt to outdo the first movie. The idea of
growing, of surpassing and ignoring everyone else’s limits, is the heart of this series’ heroes and their individual journeys. It looks like the movies
themselves are designed to follow suit.

Jawan

Where to watch: Available to stream on Netflix

Genre: Action thriller

Run time: 2h 50m

Director: Atlee

Cast: Shah Rukh Khan, Nayanthara, Vijay Sethupathi

The biggest Indian movie of the year has landed on Netflix.
</item 76>

<item 77>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
As with the DualSense deal above, this discount applies to various colors. Typically, we see the gamepad retail between $50 and $60 depending on the colorway.
The Series X/S controller doesn't have the DualSense's advanced haptics and requires AA batteries for power by default — we recommend grabbing a rechargeable
set — but some will always find its shape and asymmetrical joysticks more comfortable. It's also simpler to set up and use on PC.

PS5 + Marvel’s Spider-Man 2 bundle

If you or someone you love hasn’t gotten their hands on a PS5 yet, a new Black Friday bundle pairs the $500 console with the new (and critically acclaimed)
Spider-Man 2 game for no extra cost. The PS5 remains one of the best gaming consoles you can get right now, and we found the open-world game to be even better
than its predecessor in our review. In it, you can swap between playing as Peter Parker and Miles Morales, and it includes expanded combat mechanics.
</item 77>

<item 78>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

As Thanksgiving fast approaches, Americans are in the midst of writing their menus and buying the ingredients for their lavish feasts. And although it’s a day
filled with food and family, there is another popular tradition that’s held around the holiday: Friendsgiving.

Every year, Thanksgiving is celebrated on the fourth Thursday of the month, with the holiday falling on 23 November this year.
</item 78>

<item 79>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Of the two Spider-Mans, it’s Miles who feels more alive with the possibility to break free of expectation and canon

Narratively, juggling two Spider-Mans requires a different kind of deftness. Two villains new to the series take center stage — Kraven the Hunter and Venom — as
well as a returning cast of nearly every villain from the previous two games. Without delving into spoilers, a large majority of that cast is sidelined early
on, leaving Kraven and Venom as our principal antagonists, albeit with large roles for returning villains the Lizard (who made a brief appearance in Miles
Morales) and Mister Negative. There are others besides them that should delight comic fans, setting up future conflicts for either expansions or the presumed
Spider-Man 3 to tackle. If what you come to Spider-Man for is spectacle and villainy, Spider-Man 2 has the goods and then some.

As for our heroes, Miles and Peter return from their individual outings for a dramatic team-up.
</item 79>

<item 80>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
But feelings of loneliness spike during the holidays, meaning that there’s a good chance you’ll feel even more like Peter Parker when late November rolls
around.

It’s what makes the Thanksgiving scene in the 2002 Spider-Man film so engaging. That whole movie is an exercise in heart-on-your-sleeve sweetness. Everyone
who’s seen it knows the Thanksgiving dinner rapidly descends into chaos, with Norman Osborn ominously sticking his fingers into Aunt May’s sweet-potato
casserole, figuring out Spider-Man’s secret identity, and delivering a grossly sexist diatribe to his son Harry, leaving Harry and Mary Jane at odds with each
other. Aunt May and Peter clearly wind up with a ton of leftovers after everyone else storms out.

But for Peter, who’s just lost his Uncle Ben and has been facing the initial trials of being Spider-Man, there’s a nice moment of personal respite at the
opening of that scene, where he walks into a room where he knows he’s loved, bearing an offering of cranberry sauce,.
</item 80>

<item 81>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.
</item 81>

<item 82>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
But he makes great use of expressive close-ups (often of himself) and shallow focus, with a few creepy It Follows-like shots of blurry figures approaching from
the distance, and a terrifically claustrophobic scene inside an MRI machine. Dalton’s college story, meanwhile, occasionally borders on campus-prank zaniness:
It includes what can only be described as a puke ghost, and there’s one amusing use of the horror movie cliche about the haunted little kid who makes terrifying
drawings of the ghouls only he can see. (Naturally, that kid grows up to become a star pupil in an insufferable freshman art class.)
</item 82>

<item 83>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Thanksgiving Thursday, when stores in the U.S. are closed and many are spending time off work, has become the de facto start of the holiday shopping season both
for those looking to get bargains online and for online retailers to kick off holiday sales deals to meet that demand. But if there is a message from this
year’s Thanksgiving sales, it is that consumers are holding steady, but don’t hold your breath for a growth boom.

Adobe Analytics said that people in the U.S. on Thursday spent $5.6 billion online, which it calculates at an increase of just 5.5% on last year.

Salesforce, which also calculates sales based on data collected by its Commerce Cloud division, noted that globally, online sales reached $31.7 billion with its
U.S. tally at $7.5 billion — both up only 1%. Salesforce’s calculations of average order value were equally modest.
</item 83>

<item 84>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
MJ in Spider-Man 2 vs. MJ from Insomniac’s 2018 installment

While there are similarities between the character’s iterations in two titles, a lot of drastic differences are also present. Speaking of similar aspects, the
first game had a mission where Mary gets to use a taser to shock the guards. Well there’s no taser in Spider-Man 2, she’s got a stun gun now.

Then, there are moments of hiding and sneaking past the enemies. In the original game, there were missions like the one in Grand Central Station where MJ had to
stay hidden from a number of Mr. Negative’s demons to defuse a bomb. Similarly, in Spider-Man 2, MJ is seen hiding from the hunters during the ‘Wake Up’ quest.
But what’s actually changed?

Well, the very difference lies in the lack of MJ’s abilities in the first game and how she’s evolved in the second.
</item 84>

<item 85>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The game also sheds its predecessor’s odious Spider-Cop bits in favor of establishing a deeper connection between the Spider-Mans and their city, giving the
game room to breathe while further immersing you in its heroes’ worlds and struggles.

And then there’s the gameplay. For my money, there are few experiences more enjoyable in gaming than swinging around New York City with Spider-Man. It was fun
in the early 2000s, and it’s still fun now. And with a bigger map to explore (and wings to fly, if you so choose to use them), there’s basically no limit to the
fun you can get up to. The deployment of two protagonists is seamless — Peter and Miles play differently, bringing their own stories and desires to the table —
and switching between them is effortless. It’s also the rare open-world game that doesn’t feel bloated with missions and side quests, instead leaving you
wanting more Spider adventures. —PV

7.
</item 85>

<item 86>
Dropout’s dropping CollegeHumor on its 5th anniversary — here’s why and what the future holds (Polygon, 2023-09-26T17:35:00+00:00)
If you have a sober eye and look around at the entertainment landscape at other places that are able to celebrate improv and celebrate spontaneous comedy,
whether on Game Changer or Make Some Noise and things that push the envelope and play with the form — or even, one of my favorite things to do with Dimension 20
is taking improv comedians, who their entire creative life has been around following game, comedic beats, very in that UCB sketch structure, and let them open
their narrative wings and really dip into storytelling in a way they often don’t get an opportunity to do on stage. So on the one hand, yes, very much impostor
syndrome, and Oh, something that nice can’t be true. And then on the other hand, I do recognize the rare air we are breathing with having a space like this, so
that’s really wonderful.

I wasn’t completely certain that you’d slipped some stealth reveals into the announcement video, but I figured it was pretty likely. Which of them are you most
looking forward to?
</item 86>

<item 87>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
The length of the race can vary, with options ranging from a 5K to a half marathon, so you and your pals can take your pick at what you’d prefer.

For example, in New York City this year, one Turkey Trot is being held at the Flushing Meadows-Corona Park. During the race, which takes place on 25 November,
participants will be running for the Rolling Thunder organsiation, which is a non-profit “for mainstream athletes who are developmentally disabled or have other
special needs”.

At-home wine tasting

Typically, a night out at the bars with your friends can get a bit expensive, depending on which spots you go to. In a similar vein, a trip to a winery can rack
up a hefty bill, and it’s not the most enjoyable activity when it’s cold outside. With that in mind, your Friendsgiving plans could consist of hhosting your own
at-home wine tasting.
</item 87>

<item 88>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
At the risk of diminishing its accomplishments, I’m going to make a comparison that should cut to the quick for interested parties: This is the most A Short
Hike-alike that I’ve discovered yet, and I’ve been chasing that singular high for years. So if you have a lazy weekend afternoon and want to spend it playing —
not gaming, but playing, in the joyous, unstructured sense of a day in the park — I can’t recommend this li’l game enough. —Chris Grant

Related Lil Gator Game has the heart of Wind Waker and the adventure of Breath of the Wild

34. Starfield

Developer: Bethesda Game Studios

Where to play: Windows PC and Xbox Series X

Starfield had so much to prove, it’s easy to lose sight of what it accomplished: top-tier world-building, a fantastically customizable ship builder, and
Bethesda’s most engaging combat to date.
</item 88>

<item 89>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
New on Peacock

My Big Fat Greek Wedding 3

Where to watch: Available to stream on Peacock

Genre: Romantic comedy

Run time: 1h 32m

Director: Nia Vardalos

Cast: Nia Vardalos, John Corbett, Louis Mandylor

One of cinema’s most endearingly goofy families is back, in the most family-centric franchise this side of the Fast and Furious movies. It’s the first Greek
Wedding movie since 2016, which came nearly 15 years after the original smash hit. This time, star-writer Nia Vardalos takes over directorial duties, following
up her 2009 directorial debut I Hate Valentine’s Day.
</item 89>

<item 90>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

Making a collaborative list of the year’s best movies usually feels like trying to form a committee to definitively rank apples and oranges. How do you account
for personal tastes? For access, when a given movie — er, fruit variety — is only available in certain markets at certain times? For the fact that apples and
oranges are just so qualitatively different that they don’t have a lot in common except juiciness and vitamin C content?

Fortunately, 2023 has its own peculiar answer, in the form of the Barbenheimer phenomenon.
</item 90>

<item 91>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
What is Thanksgiving and where did it come from?

Thanksgiving is a national holiday in the United States celebrated every fourth Thursday in November.

The annual feast is in honour of the “first” Thanksgivings in America, in 1619 in Virginia, and in 1621, when colonists in Plymouth, Massachusetts, later known
as the Pilgrims, allegedly shared a meal with the Wampanoag Indians, who were native to the land.

The later feast was in honour of the help the Pilgrims received from the Native Americans in cultivating crops and surviving their first harsh winter and it
lasted three days.

Macy's Thanksgiving Day Parade takes place in New York Show all 15 1 / 15 Macy's Thanksgiving Day Parade takes place in New York Macy's Thanksgiving Day Parade
takes place in New York Performers walk in front of Macy's Tom Turkey float in the Thanksgiving Day Parade in New York AP Macy's Thanksgiving Day Parade takes
place in New York The Ronald McDonald balloon is inflated Getty Macy's Thanksgiving Day Parade takes place in New York Sinclair's DINO balloon flies as the
</item 91>

<item 92>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Without plot filling the film, we have time to just be present in this beautiful Pacific Northwest scene.

Reichardt’s genius is getting the audience giggling at the artists but never the art. For example, it’s funny to think that an artist dedicated a year of her
life to crocheting a jumpsuit. Except then, in Showing Up, you see the outfit and it’s beautiful — an intentional undermining of the punchline. A teacher smugly
opines on ceramics, but each piece he holds up is so lovingly crafted that they confidently speak for themselves.

This decision (rib artists, celebrate art) sets the tone. We humans are artifice, a bunch of contradictory masks that we put on to match the situation and the
crowd. But our creations — when we commit to a craft, whatever medium it may be — are an expression of our most vulnerable selves.
</item 92>

<item 93>
Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit (FOX News - Lifestyle, 2023-11-23T08:50:55+00:00)
"They’ve been part of a fabric of our region from Indigenous people until today."

Urann saw what cranberries could be far beyond the land of the Pilgrims.

"I felt I could do something for New England," Urann told the Associated Press in a story that ran around the nation on Thanksgiving in 1959.

CLICK HERE TO GET THE FOX NEWS APP

"You know, everything in life is what you do for others."

To read more stories in this unique "Meet the American Who…" series from Fox News Digital, click here .
</item 93>

<item 94>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Sly

Where to watch: Available to stream on Netflix

Genre: Documentary

Run time: 1h 35m

Director: Thom Zimny

Cast: Sylvester Stallone, Arnold Schwarzenegger, Quentin Tarantino

This documentary takes a close look at the life of one of the great American movie stars and film writers: Sylvester Stallone.

Nyad

Where to watch: Available to stream on Netflix

Genre: Biographical sports drama

Run time: 2h 1m

Directors: Jimmy Chin, Elizabeth Chai Vasarhelyi

Cast: Jodie Foster, Annette Bening, Rhys Ifans

Nyad tells the (questionably) true story of swimmer Diana Nyad, who swam from Cuba to Florida at 64 years old, among many other swimming accomplishments.
</item 94>

<item 95>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
New on AMC Plus

Sympathy for the Devil

Where to watch: Available to stream on AMC Plus

Genre: Psychological thriller

Run time: 1h 30m

Director: Yuval Adler

Cast: Nicolas Cage, Joel Kinnaman

A largely two-person movie that sounds a bit like Collateral, Sympathy for the Devil stars Nicolas Cage as a passenger who holds a driver (Joel Kinnaman)
hostage on a long car trip.

New to rent

Paw Patrol: The Mighty Movie

Where to watch: Available to rent on Amazon, Apple, and Vudu

Genre: Action adventure

Run time: 1h 35m

Director: Cal Brunker

Cast: Mckenna Grace, Taraji P. Henson, Marsai Martin

The Paw Patrol is back — this sequel to the first movie sees the pup get superpowers in their quest to stop Mayor Humdinger from destroying Adventure City.
</item 95>

<item 96>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
(If you can get past the Frankenstein-ification of the boroughs, that is. RIP Chelsea.)

There are hints of greater thematic aspirations within Spider-Man 2. Questions about mass incarceration, recidivism, and reintegration into society for formerly
incarcerated people; worries about the surveillance state and what we give up when we engage with modern technology. These things appear, however fleetingly, in
Spider-Man 2, but there isn’t enough time to elaborate on any single one in the unrelenting march of set-piece after glorious set-piece.

But I’m glad they’re there, anyway. Because if we’re going to retell a story, we ought to retell it anew. Canons were made to be broken, and Spider-Man 2 swings
in that direction. With a sequel teed up by the game’s final act, I’m eager to see where Insomniac goes with that momentum. Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct.
</item 96>

<item 97>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Mary’s gameplay missions end up increasing the overall quality of Spider-Man 2. All things considered, it is great to see Insomniac improving on this aspect
instead of removing it completely.

ADVERTISEMENT Article continues below this ad

Marvel’s Spider-Man 2 is now available on PlayStation 5 worldwide.

Watch This Story: The Console Wars Are Heating Head-to-Head: Xbox and PlayStation Fans Battle Over Which Custom Controller is Better, Starfield or Spider-Man 2?
</item 97>

<item 98>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Happy Friday, Polygon readers! Each week, we round up the most notable releases to streaming and video rental, highlighting the biggest and best new movies for
you to watch at home.

This week’s biggest debut is Spider-Man: Across the Spider-Verse, which is now streaming on Netflix. That’s not all, as Insidious: The Red Door — the fifth
installment in the Insidious horror franchise — also arrives on the platform this week alongside Jawan, one of the biggest Indian action movies of the year.
There’s plenty more exciting releases this week too, with A Haunting in Venice now streaming on Hulu, the Italian superhero movie Freaks vs. the Reich on Prime
Video, plus the premiere of The Kill Room and Outlaw Johnny Black from director-star Michael Jai White on VOD.

Here’s everything new to watch this weekend!
</item 98>

<item 99>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Or, to give a more mass media example, like Star Wars: Visions, which gives creative leeway to a variety of animators and storytellers to play with the very
basics of what makes Star Wars Star Wars.

Spider-Man 2 doesn’t go quite this far, but where it does go is still satisfyingly slant. Wrinkles to critical characters and plot points will keep even
seasoned Spider-fans on their toes. Moreover, it plays like a dream, with smart additions to combat that enhance the power fantasy of being Spider-Man. Each
Spider-Man now has a full arsenal of special moves and gadgets that complement one another, making even the most basic combat encounter feel like a
choreographed ballet of kicks, thwips, and zaps. The game is beautifully rendered, taking full advantage of the newest PlayStation, with such detailed textures
on each unlockable suit that you’d swear they were real, and ray-traced window after ray-traced window in a loving rendition of New York.
</item 99>

<item 100>
Thanksgiving 2023: What stores will be open on the holiday? (The Independent - Life and Style, 2023-11-06T10:40:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

On Thursday 23 November, Americans will be hard at work to prepare their Thanksgiving dinner, with foods on the menus ranging from a roasted turkey to a green
bean casserole.

Aside from the dishes served in the lavish feast, which can vary based on families’ cultures and traditions, Thanksgiving is also the annual holiday for people
to commemorate all that they’re grateful for.

In the midst of planning the big meal, there’s also a huge range of ingredients that need to be purchased at your local grocery store.
</item 100>
=== END ITEM id=34cac88e92a5c73e ===

=== ITEM id=b1ce9a88e413d85f ===
Gold sentences and facts:
Sentence 0: On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential ground invasion.
  Fact 0.0: On October 12, 2023, Israel warned 1.1 million Gaza residents to leave for the southern region.
  Fact 0.1: The warned Gaza residents were in the northern section of the enclave.
  Fact 0.2: Israel issued the warning in advance of a potential ground invasion.
Sentence 1: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 1.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.

Retrieved items:
<item 1>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
The site offers frequent odds boosts and other bonuses, and new sign-ups also get to join the famed Caesars Rewards program that can earn them more site bonuses
among other benefits.

Here are details explaining how the Caesars Sportsbook welcome offer works:

When registering your new Caesars Sportsbook account, enter the bonus code NEWS1000 .

. Place a first bet within 30 days and if it loses, receive back the equivalent amount in the form of a Bonus Bet worth up to $1,000 .

. After that first bet has been settled, you’ll see the Bonus Bet in your account within two days .

. Use your Bonus Bet within 14 days or else it will expire.

or else it will expire. If you win with your Bonus Bet, you only receive the winnings and not the bet amount back.

BetRivers Vermont promo code

BetRivers Sportsbook is another site that has launched in many states and could be among those coming to Vermont. BetRivers has a lot to offer and many have
found it a great choice elsewhere.
</item 1>

<item 2>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Bet CBK games here: $1,550 in bonus bets with BetMGM bonus code TSN50

Bet365 bonus code TSNCO for First Bet Safety Net up to $1,000 or Bet $5, Get $150 in bonus bets for ACC-SEC Challenge games

New customers at Bet365 have their choice of promo offers when using Bet365 bonus code TSNCO. The First Bet Safety Net protects your initial wager up to $1,000!

You can also choose the Bet $5, Get $150 in bonus bets offer. Both of these promos can be used to bet on any CBK games including tonight's ACC-SEC Challenge
matchups!

Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.
</item 2>

<item 3>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
DraftKings | BetMGM vs. FanDuel | Caesars vs. BetMGM
</item 3>

<item 4>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking our Caesars Sportsbook promo code link and sign up for a new account. If your first bet loses, Caesars will credit your account with bonus
bets in the amount of your wager up to $1,000.

Click to claim: Caesars Sportsbook promo code NEWS1000 for First Bet up to $1K on Caesars

BetMGM bonus code TSN50 unlocks $1,550 in bonus bets for Blue Devils vs. Razorbacks

Use our exclusive BetMGM bonus code TSN50 to claim a First Bet bonus of up to $1,500, and a $50 bonus bet for customers in select states.

Click our BetMGM bonus code link and register for your new account with promo code TSN50 to unlock this bonus.
</item 4>

<item 5>
Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook,
DraftKings and FanDuel  (Sporting News, 2023-12-18T21:00:03+00:00)
To claim this top-notch welcome offer, click our Caesars Sportsbook promo code link and sign up for a new Caesars account. Be sure Caesars Sportsbook promo code
NEWS1000 is entered when prompted. Then make your first deposit ($10 minimum).

After that, place your first Caesars Sportsbook bet and if it loses you’ll get the amount of that wager (up to $1,000) as a single bonus bet.

Get up to $1,200 in bonuses for Eagles vs.
</item 5>

<item 6>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking on our exclusive DraftKings promo code link

Register as a new user by supplying personal info including your name, email, last four of your SSN#, and mailing address

Deposit $5 to activate $50 bonus bet

Bet $5 and Get $150 in bonus bets

To max out DraftKings' deposit match at $1,000, make a minimum deposit of $5,000 and meet the 25X playthrough requirement.

Claim here: DraftKings promo code for $1,200 in bonus bets

BetRivers promo code SPORTSRIV unlocks 2nd chance bet for CBK games including Duke vs. Arkansas

The BetRivers promo code SPORTSRIV gives new users a 2nd chance bet up to $500 depending on your state.

If your first bet settles as a loss, BetRivers will refund your account in bonus bets.
</item 6>

<item 7>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
This offer requires a minimum $10 deposit, but you can deposit up to the amount of your first bet to $1,500 to maximize your potential winnings. If your first
bet on the app loses, BetMGM will refund your account with bonus bets in the amount of the original wager up to $1,500.

The extra $50 bonus bet offer is not available in Washington D.C., Kentucky, Mississippi, Nevada, New York, Puerto Rico and Ontario. Customers in those areas
can use BetMGM bonus code SPORTSPICK for the $1,500 First Bet offer.
</item 7>

<item 8>
2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet (CBSSports.com, 2023-10-13T21:17:17+00:00)
4. BetMGM Sportsbook: The signup offer at BetMGM caters more to larger players, a $1500 risk free offer which will credit a customer with bonus bets in the
event their first real money wager loses. The boosts and risk free offers on the site cater to smaller players as the limits for those tokens are generally less
than the other sites. With a friendly interface and a rewards program that can be redeemed for bonus bets or MGM Resort credit, this sportsbook is one of the
best in the business for a reason.

5. Caesars Sportsbook: Similar to MGM, the rewards program is what Caesars is most known for in that customers can redeem their wagering credits for use at
Caesars properties. New customers can get up to $1000 in bonus bets if they lose their first bet. The sportsbook offers an extensive betting menu and
specializes in specific boosts related to your home state teams.

6.
</item 8>

<item 9>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
7 Tennessee vs. No. 14 North Carolina. Duke vs. Arkansas is another marquee matchup on tonight's slate of CBK games.

Claim thousands in sports betting bonuses for college basketball games with these promo codes and links

Top-rated sportsbooks -- BetMGM, Bet365, FanDuel, Caesars Sportsbook, DraftKings and BetRivers -- make it worth your while when signing up as a new user to the
tune of $5K+ in bonus bets!

Click our links to activate the best sports betting bonuses in the industry and register your sportsbook accounts now!

Caesars Sportsbook promo code NEWS1000 for $1,000 First Bet on Caesars for Volunteers vs. Tar Heels

Caesars Sportsbook promo code NEWS1000 gives new users a First Bet up to $1,000 on Caesars!
</item 9>

<item 10>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.

Claim here: Bet365 bonus code TSNCO for promo of your choice

FanDuel promo code for Bet any $5 moneyline, Get $150 in bonus bets if your team wins in CBK game

FanDuel has a new offer when using our exclusive FanDuel promo code link. Bet $5 on any moneyline, and Get $150 in bonus bets if your team wins!

It's important to note that there are no odds restrictions, so it'd be prudent to bet a big favorite over an outmatched underdog on the CBK schedule. You need
to win the wager to receive $150 in bonus bets.
</item 10>
=== END ITEM id=b1ce9a88e413d85f ===

=== ITEM id=db5f827228e1b5b5 ===
Gold sentences and facts:
Sentence 0: Both PMS and mutual funds provide expert guidance in navigating volatile markets and seizing investment opportunities that help create wealth.
  Fact 0.0: PMS provides expert guidance in navigating volatile markets.
  Fact 0.1: PMS provides expert guidance in seizing investment opportunities.
  Fact 0.2: PMS helps create wealth.
  Fact 0.3: Mutual funds provide expert guidance in navigating volatile markets.
  Fact 0.4: Mutual funds provide expert guidance in seizing investment opportunities.
  Fact 0.5: Mutual funds help create wealth.
Sentence 1: Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results, especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).
  Fact 1.0: Brown said he was surprised.
  Fact 1.1: Brown was surprised that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results.
  Fact 1.2: Kroll was appointed to consider the reasonableness of Citrin Cooperman's assumptions for HSF's previous set of results.
  Fact 1.3: HSF's previous set of results runs to end of March 2023.

Retrieved items:
<item 1>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Advertisement

The hiring process included multiple rounds of evaluation, with the list of candidates becoming smaller each time. The first pass was driven purely by data,
which was then whittled down to a double-digit list Crocker was considering as of September, and then a final shortlist, which also included Tony Gustavsson,
head coach of Australia. Multiple sources confirmed both Hayes and Gustavsson flew to the U.S. for interviews.

One source who was briefed on the situation said the federation had also checked in on the availability of Sarina Wiegman, despite clear messaging from both the
English FA and Wiegman herself in August. “I’m staying out of it. I’ve heard it (from the press officer) but no, I’m with England and I’m really happy with
England, and I have a contract until 2025,” Wiegman said. A representative from her camp declined to comment for this story.
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 3>

<item 4>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
A representative from her camp declined to comment for this story.

Crocker said in September that the final interviews would include lengthy technical and tactical assessments, as well as questions to determine the candidates’
cultural fit. He and the federation stayed fairly consistent on their desired start date since the head coach role opened in August, but that became one of the
major concessions made by U.S. Soccer in selecting Hayes.

She’s headed Stateside 🇺🇸 U.S. Soccer has appointed Emma Hayes as the 10th full-time head coach in #USWNT history » https://t.co/sR7qfSnWVx Welcome, Emma!
pic.twitter.com/WYn6Sg9RmX — U.S. Women's National Soccer Team (@USWNT) November 14, 2023

Hayes will remain exclusively with Chelsea through the end of their WSL campaign and the Champions League season. She will not work with the U.S. in
international windows.
</item 6>

<item 7>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 10>
=== END ITEM id=db5f827228e1b5b5 ===

=== ITEM id=c3efdab4b29d1719 ===
Gold sentences and facts:
Sentence 0: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 0.0: SBF, as he is known, has pleaded not guilty to some seven charges.
  Fact 0.1: The charges SBF pleaded not guilty to are charges of fraud and conspiracy.
Sentence 1: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 1.0: Judge Lewis Kaplan intervened to ask whether Bankman-Fried had ever been told by Yedidia about that money.
  Fact 1.1: The question asked whether Bankman-Fried had been told about that money in words or in substance.
  Fact 1.2: Bankman-Fried did not admit he had been told about that money until Judge Lewis Kaplan intervened.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:

=== END ITEM id=c3efdab4b29d1719 ===
